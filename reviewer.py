"""One bounded Claude review stage with optional follow-up research."""

import asyncio
import json
import logging
import os
from time import perf_counter

import stamina
from langchain_core.utils.function_calling import convert_to_openai_tool

import config
from debug import save_output
from schemas import ToolResult
from tools import tools
from utils import retry_http_error, search_session

logger = logging.getLogger(__name__)


# --- Review requests ---
# Existing tool schemas -> Claude messages -> retained review steps


@stamina.retry(on=retry_http_error, attempts=4)
async def _request(client, api_key, payload):
    """Send a Claude request, retrying transient HTTP failures.

    Args:
        client: HTTP client shared with the active research session.
        api_key: Anthropic credential for this request.
        payload: Model settings, messages, and tool definitions.

    Returns:
        The decoded Claude response after HTTP status validation.
    """
    response = await client.post(
        config.ANTHROPIC_API,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json=payload, timeout=config.REVIEW_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


async def _call_tool(call, index, available):
    """Execute a reviewer tool request and format its result for Claude.

    Args:
        call: Tool-use block containing its ID, name, and input arguments.
        index: One-based call number used in saved artifacts.
        available: Mapping of allowed tool names to their implementations.

    Returns:
        A tool-result block containing serialized resources or an error.
        External cancellation propagates instead of becoming a result.
    """
    name, inputs, started = call["name"], call["input"], perf_counter()
    logger.info("ClaudeReviewer | %s started", name)
    try:
        tool = available[name]
        async with asyncio.timeout(config.TOOL_TIMEOUTS.get(name, config.TOOL_TIMEOUT)):
            results = await tool.ainvoke(inputs)
        if isinstance(results, str):
            raise ValueError(results)
        result = ToolResult(tool=name, query=inputs.get("query", inputs.get("keywords", "")),
                            status="success", results=results)
    except Exception as exc:
        result = ToolResult(tool=name, query=str(inputs), status="error",
                            error=f"{type(exc).__name__}: {exc}")
    except BaseException as exc:
        save_output(f"review/tools/{index}.json", {
            "tool": name, "inputs": inputs, "status": "error", "error": type(exc).__name__,
            "elapsed_seconds": perf_counter() - started,
        })
        raise
    result.elapsed_seconds = perf_counter() - started
    save_output(f"review/tools/{index}.json", result)
    logger.info("ClaudeReviewer | %s %s (%d entries, %.2fs)",
                name, result.status, len(result.results), result.elapsed_seconds)
    return {"type": "tool_result", "tool_use_id": call["id"],
            "content": result.model_dump_json(), "is_error": result.status == "error"}


# --- Review workflow ---
# Draft + evidence -> up to three tool calls -> revised answer or original draft


async def review_answer(question: str, findings: dict, draft: str) -> str:
    """Review a draft within a total deadline and a bounded tool-call budget.

    Args:
        question: Original user question used to assess the draft's relevance.
        findings: Specialist results, including their supporting tool evidence.
        draft: Existing answer to revise or retain as a fallback.

    Returns:
        Revised Markdown, or the original draft when credentials are missing
        or review fails. External cancellation still propagates.
    """
    started = perf_counter()
    record = {"model": config.REVIEW_MODEL, "status": "skipped", "tool_calls": 0, "answer": draft}
    try:
        async with asyncio.timeout(config.REVIEW_TIMEOUT):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                record["reason"] = "ANTHROPIC_API_KEY is not configured."
                logger.warning("ClaudeReviewer skipped: ANTHROPIC_API_KEY is not configured")
                return draft
            logger.info("ClaudeReviewer started (max %d tool calls, %ss)",
                        config.REVIEW_MAX_TOOL_CALLS, config.REVIEW_TIMEOUT)
            # Reuse provider wrappers so review follows the same tool boundaries.
            available = {tool.name: tool for tool in tools}
            definitions = []
            for tool in available.values():
                schema = convert_to_openai_tool(tool)["function"]
                definitions.append({"name": schema["name"], "description": schema["description"],
                                    "input_schema": schema["parameters"]})
            messages = [{"role": "user", "content": json.dumps({
                "question": question, "findings": findings, "draft": draft,
            })}]
            async with search_session() as client:
                # Each tool-using response must consume a slot; the last response must finish.
                for step in range(config.REVIEW_MAX_TOOL_CALLS + 1):
                    exhausted = record["tool_calls"] >= config.REVIEW_MAX_TOOL_CALLS
                    response = await _request(client, api_key, {
                        "model": config.REVIEW_MODEL, "max_tokens": 8192,
                        "system": config.REVIEW_SYSTEM_PROMPT, "messages": messages,
                        "tools": definitions,
                        "tool_choice": {"type": "none"} if exhausted else {
                            "type": "auto", "disable_parallel_tool_use": True,
                        },
                    })
                    save_output(f"review/steps/{step + 1}.json", response)
                    content = response["content"]
                    calls = [block for block in content if block["type"] == "tool_use"]
                    if calls and response.get("stop_reason") != "tool_use":
                        raise ValueError("Claude returned incomplete tool requests.")
                    if not calls:
                        answer = "\n".join(block["text"] for block in content if block["type"] == "text").strip()
                        if response.get("stop_reason") != "end_turn" or not answer:
                            raise ValueError("Claude did not return a complete review answer.")
                        record.update(status="success", answer=answer)
                        return answer
                    messages.append({"role": "assistant", "content": content})
                    results = []
                    for call in calls:
                        if record["tool_calls"] < config.REVIEW_MAX_TOOL_CALLS:
                            # Failed calls consume a slot too, keeping retries bounded.
                            record["tool_calls"] += 1
                            results.append(await _call_tool(call, record["tool_calls"], available))
                        else:
                            results.append({"type": "tool_result", "tool_use_id": call["id"],
                                            "content": "Tool budget exhausted. Finish with available evidence.",
                                            "is_error": True})
                    messages.append({"role": "user", "content": results})
                raise ValueError("Claude did not finish within the review tool budget.")
    except Exception as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        logger.warning("ClaudeReviewer failed (%s); keeping master draft", type(exc).__name__)
        return draft
    except BaseException as exc:
        record.update(status="error", error=type(exc).__name__)
        raise
    finally:
        record["elapsed_seconds"] = perf_counter() - started
        save_output("review/result.json", record)
        logger.info("ClaudeReviewer %s (%d tool calls, %.2fs)",
                    record["status"], record["tool_calls"], record["elapsed_seconds"])
