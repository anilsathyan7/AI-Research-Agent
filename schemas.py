"""Shared planning models, search resources, and agent findings."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# --- Level 1: Planning output (tool queries and agent routing) ---
# Question -> agent routing -> tool-specific queries


class ToolQuery(BaseModel):
    """A provider-specific query paired with its tool name."""

    tool: str
    query: str = Field(min_length=1, pattern=r"\S")


class PreparedQueries(BaseModel):
    """Queries prepared together for an agent's search tools."""

    queries: list[ToolQuery]


class ResearchTask(BaseModel):
    """A focused research query assigned to one agent by the master."""

    agent: Literal["dataset", "model", "paper", "code"]
    query: str = Field(min_length=1, pattern=r"\S")


class RelevanceChecklist(BaseModel):
    """Question-specific guidance for selecting and qualifying evidence."""

    include: list[str]
    exclude_or_label: list[str]


class ResearchPlan(BaseModel):
    """The master's routing plan, selecting up to four distinct agents."""

    tasks: list[ResearchTask] = Field(min_length=1, max_length=4)
    checklist: RelevanceChecklist

    @model_validator(mode="after")
    def unique_agents(self):
        """Validate that each specialist is dispatched at most once.

        Returns:
            This research plan after validation.

        Raises:
            ValueError: More than one task selects the same agent.
        """
        if len({task.agent for task in self.tasks}) != len(self.tasks):
            raise ValueError("Select each research agent at most once.")
        return self


# --- Level 2: Tool output ---
# Provider response -> evidence and metadata -> ToolResult


class Evidence(BaseModel):
    """Supporting text labeled as an abstract, snippet, README, or description."""

    type: str
    text: str


class Source(BaseModel):
    """A tool's source URL, supporting evidence, and original-response reference."""

    tool: str
    url: str
    evidence: list[Evidence]
    raw_ref: str | None = None


class Resource(BaseModel):
    """A discovered resource with metadata and source-attributed evidence."""

    id: str
    kind: Literal["paper", "dataset", "model", "repository", "webpage"]
    title: str
    url: str
    metadata: dict = Field(default_factory=dict)
    sources: list[Source]


class ToolResult(BaseModel):
    """One tool call's query, results, status, timing, and error."""

    tool: str
    query: str
    status: Literal["success", "error"]
    elapsed_seconds: float = 0
    results: list[Resource] = Field(default_factory=list)
    error: str | None = None


# --- Level 3: Agent output ---
# ToolResult -> AgentSynthesis -> AgentResult -> master answer


class Finding(BaseModel):
    """A supported conclusion linked to resource IDs, relevance, and limitations."""

    summary: str = Field(min_length=1)
    resource_ids: list[str] = Field(min_length=1)
    relevance: str
    limitations: list[str]


class AgentSynthesis(BaseModel):
    """LLM-generated findings referencing supplied resource IDs."""

    findings: list[Finding]
    limitations: list[str]


class AgentResult(AgentSynthesis):
    """Python-assembled output passed to the master with tool results intact."""

    agent: str
    query: str
    tool_results: list[ToolResult]
