"""Specialist research graphs and the master routing, synthesis, and review flow."""

import argparse
import asyncio
import json
import logging
import operator
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.types import Send

import config
from debug import capture_run, save_output
from reviewer import review_answer
from schemas import AgentResult, AgentSynthesis, PreparedQueries, RelevanceChecklist, ResearchPlan, ResearchTask, Source, ToolResult
from utils import call_llm, enrich_url, raw_responses, search_session
from tools import ai_search, search_arxiv, search_gemini, search_github, search_github_code, search_huggingface, search_kaggle, search_kaggle_models, search_models, search_openalex, search_serper, tools as research_tools

logger = logging.getLogger(__name__)


# --- Shared research workflow ---
# Prepare queries -> parallel searches -> enrichment -> structured findings


class ResearchState(MessagesState):
    """Shared specialist state for prepared queries, evidence, and synthesis."""

    queries: dict[str, str]
    checklist: RelevanceChecklist | None
    gemini_results: ToolResult
    result: AgentResult


class ResearchAgent(ABC):
    """Run a specialist's searches, page enrichment, and evidence synthesis."""

    state_schema: type[ResearchState]
    agent_name: str
    result_fields: tuple[str, ...]

    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        system_prompt: str = config.SYSTEM_PROMPT,
    ) -> None:
        """Initialize shared state; subclasses supply their tools and prompt.

        Args:
            tools: Provider tools to include in query preparation.
            system_prompt: Specialist instructions for evidence synthesis.
        """
        self.tools = research_tools if tools is None else tools
        self.system_prompt = system_prompt
        self.llm = ChatOpenAI(**config.MODEL_CONFIG)
        self.graph = self._build_graph()

    @abstractmethod
    async def search_text(self, state: MessagesState) -> dict:
        """Run this agent's text-search providers.

        Args:
            state: Specialist state containing prepared provider queries.

        Returns:
            State updates containing the branch's tool results.
        """

    @abstractmethod
    async def search_semantic(self, state: MessagesState) -> dict:
        """Run this agent's semantic-search providers.

        Args:
            state: Specialist state containing prepared provider queries.

        Returns:
            State updates containing the branch's tool results.
        """

    @abstractmethod
    async def search_web(self, state: MessagesState) -> dict:
        """Run this agent's web-search providers.

        Args:
            state: Specialist state containing prepared provider queries.

        Returns:
            State updates containing the branch's tool results.
        """

    @abstractmethod
    async def search_ai(self, state: MessagesState) -> dict:
        """Run this agent's AI-search providers.

        Args:
            state: Specialist state containing prepared provider queries.

        Returns:
            State updates containing the branch's tool results.
        """

    async def prepare_queries(self, state: ResearchState) -> dict:
        """Prepare every tool's query in one LLM call.

        Args:
            state: Specialist state whose first message contains the question.

        Returns:
            A queries update mapping each tool name to its prepared query.

        Raises:
            ValueError: The plan contains missing, duplicate, or unexpected tools.
        """
        if not self.tools:
            return {"queries": {}}

        name, started = type(self).__name__, perf_counter()
        logger.info("%s | preparing queries", name)
        # Prepare every provider's query together using its own guidance.
        response = await call_llm(self.llm.with_structured_output(PreparedQueries), [
            SystemMessage(content=config.QUERY_PREPARATION_PROMPT),
            HumanMessage(content=json.dumps({
                "agent": name,
                "question": state["messages"][0].content,
                "tools": [{"name": tool.name, "instructions": tool.description} for tool in self.tools],
            })),
        ])
        save_output(f"{self.agent_name}/queries.json", response)
        # Reject missing, duplicate, or unexpected tool assignments before searching.
        plan = PreparedQueries.model_validate(response)
        queries = {item.tool: item.query for item in plan.queries}
        if len(queries) != len(plan.queries) or set(queries) != {tool.name for tool in self.tools}:
            raise ValueError("Prepare exactly one query for each requested tool.")
        logger.info("%s | queries prepared (%d tools, %.2fs)", name, len(queries), perf_counter() - started)
        return {"queries": queries}

    def _build_graph(self):
        """Build the fixed search, enrichment, and synthesis workflow.

        Returns:
            A compiled graph that waits for all searches before enrichment.
        """
        builder = StateGraph(self.state_schema)
        builder.add_node("prepare_queries", self.prepare_queries)
        builder.add_edge(START, "prepare_queries")
        # All search branches start after query preparation finishes.
        searches = ["search_text", "search_semantic", "search_web", "search_ai"]
        for name in searches:
            builder.add_node(name, getattr(self, name))
            builder.add_edge("prepare_queries", name)
        builder.add_node("enrich_results", self.enrich_results, input_schema=self.state_schema)
        builder.add_node("combine_results", self.combine_results, input_schema=self.state_schema)
        # Wait for every search before enriching and synthesizing its evidence.
        builder.add_edge(searches, "enrich_results")
        builder.add_edge("enrich_results", "combine_results")
        builder.add_edge("combine_results", END)
        return builder.compile()

    def run(self, question: str) -> AgentResult:
        """Run from synchronous code; async callers should await arun().

        Args:
            question: Research question assigned to this specialist.

        Returns:
            Structured findings, limitations, and original tool results.
        """
        return asyncio.run(self.arun(question))

    async def arun(self, question: str, checklist: RelevanceChecklist | None = None) -> AgentResult:
        """Run specialist research within the active or a standalone session.

        Args:
            question: Research question assigned to this specialist.
            checklist: Optional shared relevance criteria for synthesis.

        Returns:
            Structured findings, limitations, and original tool results.

        Raises:
            ValueError: Query preparation or synthesis fails validation.
            TimeoutError: An LLM stage exceeds its deadline.
        """
        with config.log_run():
            name, started = type(self).__name__, perf_counter()
            logger.info("%s started", name)
            try:
                # Reuse the master's session, or own one for a standalone run.
                async with search_session():
                    self.raw_responses = raw_responses()
                    with capture_run(question, self.raw_responses, f"{self.agent_name}/result.json") as self.run_directory:
                        state = await self.graph.ainvoke({"messages": [("user", question)], "checklist": checklist})
                        result = AgentResult.model_validate(state["result"])
                        save_output(f"{self.agent_name}/result.json", result)
            except Exception as exc:
                logger.error("%s failed (%s, %.2fs)", name, type(exc).__name__, perf_counter() - started)
                raise
            logger.info("%s finished (%.2fs)", name, perf_counter() - started)
            return result

    async def call_tool(self, tool: BaseTool, inputs: dict) -> ToolResult:
        """Run a tool under its deadline and retain failures as results.

        Args:
            tool: Provider tool to invoke.
            inputs: Arguments matching the tool's input schema.

        Returns:
            Validated resources or an error result, with query and timing.
            External cancellation propagates instead of becoming a result.
        """
        name, started = type(self).__name__, perf_counter()
        logger.info("%s | %s started", name, tool.name)
        query = inputs.get("query", inputs.get("keywords", ""))
        exception = None
        try:
            # The deadline includes provider retries; result validation stays inside the handler.
            async with asyncio.timeout(config.TOOL_TIMEOUTS.get(tool.name, config.TOOL_TIMEOUT)):
                result = await tool.ainvoke(inputs)
            error = result if isinstance(result, str) else None
            output = ToolResult(
                tool=tool.name, query=query,
                status="error" if error is not None else "success",
                elapsed_seconds=perf_counter() - started,
                results=[] if error is not None else result, error=error,
            )
        except Exception as exc:
            # A failed provider becomes evidence of failure without stopping healthy branches.
            exception = exc
            output = ToolResult(
                tool=tool.name, query=query, status="error",
                elapsed_seconds=perf_counter() - started,
                error=f"{tool.name} failed: {type(exc).__name__}: {exc}",
            )
        except BaseException as exc:
            # Preserve cancellation details, then let cancellation propagate.
            save_output(f"{self.agent_name}/tools/{tool.name}.json", {
                "tool": tool.name, "inputs": inputs, "status": "error",
                "elapsed_seconds": perf_counter() - started,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
            raise
        if output.status == "error":
            logger.warning(
                "%s | %s failed (%.2fs)", name, tool.name, output.elapsed_seconds,
                exc_info=exception,
            )
        else:
            logger.info(
                "%s | %s finished (%d entries, %.2fs)",
                name, tool.name, len(output.results), output.elapsed_seconds,
            )
        save_output(f"{self.agent_name}/tools/{tool.name}.json", output)
        return output

    async def enrich_results(self, state: ResearchState) -> dict:
        """Append full extracted page evidence without replacing search results.

        Args:
            state: Specialist state containing all configured tool results.

        Returns:
            Copied tool-result updates with successful extractions attached.
            Failed extractions leave the original search evidence available.
        """
        name, started = type(self).__name__, perf_counter()
        # Copy results so enrichment preserves the original search artifacts.
        results = {field: ToolResult.model_validate(state[field]).model_copy(deep=True)
                   for field in self.result_fields}
        urls = list(dict.fromkeys(resource.url for batch in results.values()
                                  for resource in batch.results if resource.url))
        logger.info("%s | enriching %d URLs", name, len(urls))
        # Fetch unique URLs concurrently; enrich_url shares cached work across agents.
        pages = dict(zip(urls, await asyncio.gather(*(enrich_url(url) for url in urls))))
        # Append successful extractions; failed pages retain their search evidence.
        for batch in results.values():
            for resource in batch.results:
                page = pages.get(resource.url, {})
                if page.get("status") == "success":
                    resource.sources.append(Source.model_validate(page["source"]))
                    resource.metadata.setdefault("page_metadata", page["metadata"])
        save_output(f"{self.agent_name}/enrichment.json", pages)
        logger.info("%s | enrichment finished (%d enriched, %d skipped/failed, %.2fs)",
                    name, sum(page["status"] == "success" for page in pages.values()),
                    sum(page["status"] != "success" for page in pages.values()), perf_counter() - started)
        return results

    async def combine_results(self, state: ResearchState) -> dict:
        """Generate findings by reference, preserving all resources in Python.

        Args:
            state: Original question, tool results, and optional checklist.

        Returns:
            A result update containing findings and their supporting resources.

        Raises:
            ValueError: Synthesis is invalid or references an unknown resource.
        """
        logger.info("%s | combining results", type(self).__name__)
        tool_results = [ToolResult.model_validate(state[field]) for field in self.result_fields]
        # Ask the LLM for findings and references, not rewritten resource records.
        response = await call_llm(self.llm.with_structured_output(AgentSynthesis), [
            SystemMessage(content=self.system_prompt + config.AGENT_OUTPUT_PROMPT),
            state["messages"][0],
            HumanMessage(content=f"Relevance checklist: {json.dumps(state['checklist'].model_dump())}")
            if state.get("checklist") else HumanMessage(content="No shared relevance checklist was supplied."),
            HumanMessage(content=json.dumps([
                result.model_dump(exclude_none=True) for result in tool_results
            ])),
        ], reasoning_effort="high")
        save_output(f"{self.agent_name}/synthesis.json", response)
        synthesis = AgentSynthesis.model_validate(response)
        # Validate references before attaching the unchanged supporting resources.
        resource_ids = {resource.id for result in tool_results for resource in result.results}
        if any(set(finding.resource_ids) - resource_ids for finding in synthesis.findings):
            raise ValueError("Findings must reference supplied resource IDs.")
        return {"result": AgentResult(
            agent=self.agent_name, query=state["messages"][0].content,
            tool_results=tool_results, **synthesis.model_dump(),
        )}


# --- Dataset research ---
# Hugging Face + Kaggle datasets + Serper + Tavily + Gemini


class DatasetResearchState(ResearchState):
    """Dataset-provider and web-search results for the dataset specialist."""

    dataset_results: ToolResult
    kaggle_results: ToolResult
    web_results: ToolResult
    ai_results: ToolResult


class DatasetResearchAgent(ResearchAgent):
    """Research datasets through Hugging Face, Kaggle, and web providers."""

    state_schema = DatasetResearchState
    agent_name = "dataset"
    result_fields = ("dataset_results", "kaggle_results", "web_results", "ai_results", "gemini_results")

    def __init__(self) -> None:
        """Configure dataset providers and the dataset synthesis role."""
        super().__init__(
            tools=[search_huggingface, ai_search, search_kaggle, search_serper, search_gemini],
            system_prompt=config.DATASET_SYSTEM_PROMPT,
        )

    async def search_semantic(self, state: DatasetResearchState):
        """Return Hugging Face dataset results for the prepared semantic query."""
        results = await self.call_tool(search_huggingface, {"query": state["queries"][search_huggingface.name]})
        return {"dataset_results": results}

    async def search_web(self, state: DatasetResearchState):
        """Return Serper results for the prepared dataset web query."""
        results = await self.call_tool(search_serper, {"query": state["queries"][search_serper.name]})
        return {"web_results": results}

    async def search_ai(self, state: DatasetResearchState):
        """Return Tavily and Gemini dataset evidence from parallel searches."""
        tavily, gemini = await asyncio.gather(
            self.call_tool(ai_search, {"query": state["queries"][ai_search.name]}),
            self.call_tool(search_gemini, {"query": state["queries"][search_gemini.name]}),
        )
        return {"ai_results": tavily, "gemini_results": gemini}

    async def search_text(self, state: DatasetResearchState):
        """Return Kaggle dataset results for the prepared keyword query."""
        results = await self.call_tool(search_kaggle, {"keywords": state["queries"][search_kaggle.name]})
        return {"kaggle_results": results}


# --- Paper research ---
# arXiv + OpenAlex + Serper + Tavily + Gemini


class PaperResearchState(ResearchState):
    """Academic-provider and web-search results for the paper specialist."""

    arxiv_results: ToolResult
    openalex_results: ToolResult
    web_results: ToolResult
    ai_results: ToolResult


class PaperResearchAgent(ResearchAgent):
    """Research papers through arXiv, OpenAlex, and web providers."""

    state_schema = PaperResearchState
    agent_name = "paper"
    result_fields = ("arxiv_results", "openalex_results", "web_results", "ai_results", "gemini_results")

    def __init__(self) -> None:
        """Configure academic providers and the paper synthesis role."""
        super().__init__(
            tools=[search_arxiv, search_openalex, ai_search, search_serper, search_gemini],
            system_prompt=config.PAPER_SYSTEM_PROMPT,
        )

    async def search_text(self, state: PaperResearchState):
        """Return arXiv papers for the prepared fielded or Boolean query."""
        results = await self.call_tool(search_arxiv, {"query": state["queries"][search_arxiv.name]})
        return {"arxiv_results": results}

    async def search_semantic(self, state: PaperResearchState):
        """Return OpenAlex papers for the prepared natural-language query."""
        results = await self.call_tool(search_openalex, {"query": state["queries"][search_openalex.name]})
        return {"openalex_results": results}

    async def search_web(self, state: PaperResearchState):
        """Return Serper results for the prepared paper web query."""
        results = await self.call_tool(search_serper, {"query": state["queries"][search_serper.name]})
        return {"web_results": results}

    async def search_ai(self, state: PaperResearchState):
        """Return Tavily and Gemini paper evidence from parallel searches."""
        tavily, gemini = await asyncio.gather(
            self.call_tool(ai_search, {"query": state["queries"][ai_search.name]}),
            self.call_tool(search_gemini, {"query": state["queries"][search_gemini.name]}),
        )
        return {"ai_results": tavily, "gemini_results": gemini}


# --- Model research ---
# Hugging Face + Kaggle models + Serper + Tavily + Gemini


class ModelResearchState(ResearchState):
    """Model-provider and web-search results for the model specialist."""

    model_results: ToolResult
    kaggle_results: ToolResult
    web_results: ToolResult
    ai_results: ToolResult


class ModelResearchAgent(ResearchAgent):
    """Research pretrained models through model hubs and web providers."""

    state_schema = ModelResearchState
    agent_name = "model"
    result_fields = ("model_results", "kaggle_results", "web_results", "ai_results", "gemini_results")

    def __init__(self) -> None:
        """Configure model providers and the model synthesis role."""
        super().__init__(
            tools=[search_models, ai_search, search_kaggle_models, search_serper, search_gemini],
            system_prompt=config.MODEL_SYSTEM_PROMPT,
        )

    async def search_text(self, state: ModelResearchState):
        """Return Kaggle model results for the prepared keyword query."""
        results = await self.call_tool(search_kaggle_models, {"keywords": state["queries"][search_kaggle_models.name]})
        return {"kaggle_results": results}

    async def search_semantic(self, state: ModelResearchState):
        """Return Hugging Face models for the prepared natural-language query."""
        results = await self.call_tool(search_models, {"query": state["queries"][search_models.name]})
        return {"model_results": results}

    async def search_web(self, state: ModelResearchState):
        """Return Serper results for the prepared model web query."""
        results = await self.call_tool(search_serper, {"query": state["queries"][search_serper.name]})
        return {"web_results": results}

    async def search_ai(self, state: ModelResearchState):
        """Return Tavily and Gemini model evidence from parallel searches."""
        tavily, gemini = await asyncio.gather(
            self.call_tool(ai_search, {"query": state["queries"][ai_search.name]}),
            self.call_tool(search_gemini, {"query": state["queries"][search_gemini.name]}),
        )
        return {"ai_results": tavily, "gemini_results": gemini}


# --- Code research ---
# GitHub repositories and files + Tavily + Gemini with Google Search grounding


class CodeResearchState(ResearchState):
    """Repository, code-file, and web evidence for the code specialist."""

    github_results: ToolResult
    code_results: ToolResult
    ai_results: ToolResult


class CodeResearchAgent(ResearchAgent):
    """Research implementation evidence without executing retrieved code."""

    state_schema = CodeResearchState
    agent_name = "code"
    result_fields = ("github_results", "code_results", "ai_results", "gemini_results")

    def __init__(self) -> None:
        """Configure GitHub and web providers for implementation research."""
        super().__init__(
            tools=[search_github, search_github_code, ai_search, search_gemini],
            system_prompt=config.CODE_SYSTEM_PROMPT,
        )

    async def search_text(self, state: CodeResearchState):
        """Return GitHub repository and file evidence from parallel searches."""
        repositories, code = await asyncio.gather(
            self.call_tool(search_github, {"query": state["queries"][search_github.name]}),
            self.call_tool(search_github_code, {"query": state["queries"][search_github_code.name]}),
        )
        return {"github_results": repositories, "code_results": code}

    async def search_ai(self, state: CodeResearchState):
        """Return Tavily and Gemini implementation evidence from parallel searches."""
        tavily, gemini = await asyncio.gather(
            self.call_tool(ai_search, {"query": state["queries"][ai_search.name]}),
            self.call_tool(search_gemini, {"query": state["queries"][search_gemini.name]}),
        )
        return {"ai_results": tavily, "gemini_results": gemini}

    async def search_semantic(self, state: CodeResearchState):
        """Return no updates because no semantic code provider is configured."""
        return {}

    async def search_web(self, state: CodeResearchState):
        """Return no updates because Gemini already provides Google Search."""
        return {}


# --- Master workflow ---
# Route -> dispatch selected agents -> draft -> review -> final answer


class MasterState(TypedDict):
    """Routing, shared criteria, merged specialist results, and final answer."""

    question: str
    tasks: list[ResearchTask]
    checklist: RelevanceChecklist
    findings: Annotated[dict, operator.or_]
    answer: str


class AgentTaskState(TypedDict):
    """One dispatched specialist task and the shared relevance checklist."""

    task: ResearchTask
    checklist: RelevanceChecklist


class MasterResearchAgent:
    """Route a question to specialists, synthesize their evidence, and review it."""

    def __init__(self) -> None:
        """Initialize the router, specialist registry, and compiled workflow."""
        self.llm = ChatOpenAI(**config.MODEL_CONFIG)
        self.router = self.llm.with_structured_output(ResearchPlan)
        self.agents = {
            "dataset": DatasetResearchAgent,
            "model": ModelResearchAgent,
            "paper": PaperResearchAgent,
            "code": CodeResearchAgent,
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build routing, parallel specialist execution, synthesis, and review.

        Returns:
            A compiled graph that synthesizes after selected workers finish.
        """
        # Send creates a worker task for each selected agent.
        builder = StateGraph(MasterState)
        builder.add_node("route", self.route)
        builder.add_node("run_agent", self.run_agent)
        builder.add_node("summarize", self.summarize)
        builder.add_node("review", self.review)
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", self.dispatch_agents, ["run_agent"])
        builder.add_edge("run_agent", "summarize")
        builder.add_edge("summarize", "review")
        builder.add_edge("review", END)
        return builder.compile()

    async def route(self, state: MasterState) -> dict:
        """Select specialists and generate shared relevance criteria.

        Args:
            state: Master state containing the original user question.

        Returns:
            Validated task assignments and a relevance checklist.

        Raises:
            ValueError: The generated research plan fails validation.
        """
        started = perf_counter()
        logger.info("MasterResearchAgent | routing started")
        plan = await call_llm(self.router, [
            SystemMessage(content=config.MASTER_ROUTING_PROMPT),
            HumanMessage(content=state["question"]),
        ])
        save_output("routing.json", plan)
        # The plan schema limits dispatch to supported, distinct agents.
        research_plan = ResearchPlan.model_validate(plan)
        tasks = research_plan.tasks
        logger.info(
            "MasterResearchAgent | routing finished: %s (%.2fs)",
            ", ".join(task.agent for task in tasks), perf_counter() - started,
        )
        return {"tasks": tasks, "checklist": research_plan.checklist}

    async def dispatch_agents(self, state: MasterState) -> list[Send]:
        """Create parallel worker tasks from the validated routing plan.

        Args:
            state: Master state containing task assignments and the checklist.

        Returns:
            One graph dispatch per selected specialist, including its checklist.
        """
        logger.info(
            "MasterResearchAgent | dispatching %d agents: %s",
            len(state["tasks"]), ", ".join(task.agent for task in state["tasks"]),
        )
        return [Send("run_agent", {"task": task, "checklist": state["checklist"]}) for task in state["tasks"]]

    async def run_agent(self, state: AgentTaskState) -> dict:
        """Run one specialist while isolating its failure from other workers.

        Args:
            state: Assigned research task and shared relevance checklist.

        Returns:
            A findings update keyed by agent name, with its result or an error.
        """
        task = state["task"]
        result = {"query": task.query}
        try:
            findings = await self.agents[task.agent]().arun(task.query, checklist=state["checklist"])
            result["findings"] = AgentResult.model_validate(findings).model_dump(exclude_none=True)
        except Exception as exc:
            # Keep other agents' results available when one agent fails.
            logger.warning("%s research failed (%s)", task.agent, type(exc).__name__)
            result["error"] = f"Research failed ({type(exc).__name__})"
            save_output(f"{task.agent}/result.json", {
                "query": task.query, "status": "error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            })
        return {"findings": {task.agent: result}}

    async def summarize(self, state: MasterState) -> dict:
        """Synthesize specialist evidence into a validated master draft.

        Args:
            state: Original question, task order, checklist, and worker results.

        Returns:
            An answer update containing the completed Markdown draft.

        Raises:
            ValueError: The response is incomplete or contains no answer text.
        """
        started = perf_counter()
        # Keep routing order regardless of worker completion order.
        findings = {task.agent: state["findings"][task.agent] for task in state["tasks"]}
        logger.info(
            "MasterResearchAgent | collected %d agent results (%d failed)",
            len(findings), sum("error" in result for result in findings.values()),
        )
        logger.info("MasterResearchAgent | synthesis started")
        save_output("agents.json", findings)
        # Supply both findings and original evidence for the final synthesis.
        response = await call_llm(self.llm, [
            SystemMessage(content=config.MASTER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"User question:\n{state['question']}\n\n"
                f"Relevance checklist:\n{json.dumps(state['checklist'].model_dump())}\n\n"
                f"Research findings:\n{json.dumps(findings)}"
            )),
        ], reasoning_effort="xhigh")
        # Review may fall back to this draft, so it must already be complete.
        metadata = response.response_metadata
        if metadata.get("status") != "completed" or metadata.get("incomplete_details"):
            raise ValueError("Master synthesis did not return a complete response.")
        answer = response.text
        if not answer.strip():
            raise ValueError("Master synthesis returned an empty answer.")
        logger.info("MasterResearchAgent | synthesis finished (%.2fs)", perf_counter() - started)
        save_output("draft.json", {"answer": answer})
        return {"answer": answer}

    async def review(self, state: MasterState) -> dict:
        """Review the draft once, with bounded follow-up searches.

        Args:
            state: Original question, specialist evidence, and validated draft.

        Returns:
            An answer update containing the revision or the fallback draft.
        """
        return {"answer": await review_answer(state["question"], state["findings"], state["answer"])}

    def run(self, question: str) -> str:
        """Run from synchronous code; async callers should await arun().

        Args:
            question: Original research question from the user.

        Returns:
            The final Markdown answer, using the draft if review is unavailable.
        """
        return asyncio.run(self.arun(question))

    async def arun(self, question: str) -> str:
        """Run the full research workflow with shared sessions and artifacts.

        Args:
            question: Nonblank research question from the user.

        Returns:
            The final Markdown answer, using the draft if review is unavailable.

        Raises:
            ValueError: The question, routing plan, or final draft is invalid.
            TimeoutError: Routing or final synthesis exceeds its LLM deadline.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")
        with config.log_run():
            started = perf_counter()
            logger.info("MasterResearchAgent started")
            try:
                # Child agents inherit this run's client, caches, and debug directory.
                async with search_session():
                    self.raw_responses = raw_responses()
                    with capture_run(question, self.raw_responses, "master.json") as self.run_directory:
                        result = await self.graph.ainvoke({"question": question, "findings": {}})
                        answer = result["answer"]
                        save_output("master.json", {"answer": answer})
            except Exception as exc:
                logger.error("MasterResearchAgent failed (%s, %.2fs)", type(exc).__name__, perf_counter() - started)
                raise
            logger.info("MasterResearchAgent finished (%.2fs)", perf_counter() - started)
            return answer


# --- Terminal entry point ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research a question using specialist agents.")
    parser.add_argument("question", help="Research question to investigate.")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("The research question must not be empty.")

    config.configure_logging()
    agent = MasterResearchAgent()
    print("\n" + agent.run(args.question))

