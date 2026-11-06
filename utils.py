"""Shared LLM deadlines, HTTP sessions, page enrichment, and result normalization."""

import asyncio
import json
import logging
import socket
import sys
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
import stamina
import trafilatura
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.native_pdf_pipeline import NativePdfPipeline

import config
from schemas import Resource, Source

logger = logging.getLogger(__name__)
_http_client: ContextVar[httpx.AsyncClient | None] = ContextVar("search_http_client", default=None)
_raw_responses: ContextVar[dict | None] = ContextVar("raw_search_responses", default=None)
_page_tasks: ContextVar[dict | None] = ContextVar("page_enrichment_tasks", default=None)
_page_limit: ContextVar[asyncio.Semaphore | None] = ContextVar("page_enrichment_limit", default=None)


# --- LLM calls ---
# Messages -> response within the call deadline


async def call_llm(model, messages, **kwargs):
    """Bound an LLM invocation, including its client's retries.

    Args:
        model: Chat model or structured-output runnable with an async interface.
        messages: Conversation messages to send to the model.
        **kwargs: Invocation options such as reasoning effort.

    Returns:
        The model response, or parsed output for a structured-output runnable.

    Raises:
        TimeoutError: The invocation exceeds the configured LLM deadline.
    """
    async with asyncio.timeout(config.LLM_TIMEOUT):
        return await model.ainvoke(messages, **kwargs)


# --- HTTP retries ---
# Transient failure -> Retry-After delay or normal backoff


def retry_http_error(exc: Exception) -> bool | float:
    """Classify HTTP failures and honor valid Retry-After headers.

    Args:
        exc: Failure raised by an HTTPX request or response check.

    Returns:
        True for default backoff, a delay in seconds for Retry-After, or False
        when the failure should not be retried.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code not in {408, 429, 500, 502, 503, 504}:
            return False
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after:
            # Retry-After can be seconds or an HTTP date; invalid values use backoff.
            try:
                seconds = int(retry_after)
                return float(seconds) if seconds >= 0 else True
            except (ValueError, OverflowError):
                pass
            try:
                deadline = parsedate_to_datetime(retry_after)
                return max(0.0, (deadline - datetime.now(UTC)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
        return True
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError))


# --- Result normalization ---
# Provider records -> resources, evidence, and retained raw responses


def raw_responses() -> dict:
    """Return original responses retained for the active research run.

    Returns:
        The active session's mutable response store, or a new empty dictionary
        when no session is active.
    """
    responses = _raw_responses.get()
    return responses if responses is not None else {}


def normalize_results(tool_name: str, kind: str, records: list[dict], raw, text_types=None) -> list[dict]:
    """Normalize provider records while retaining source evidence and metadata.

    Args:
        tool_name: Provider tool responsible for these results.
        kind: Resource category accepted by the Resource schema.
        records: Provider records mapped to common title, URL, and text fields.
        raw: Original provider response to retain in the active session.
        text_types: Optional mapping from additional text fields to evidence types.

    Returns:
        Serialized resources with unique IDs and source-attributed evidence.
        Raw-response references are included only when a session is active.
    """
    call_id = uuid4().hex
    stored = _raw_responses.get()
    if stored is not None:
        stored[call_id] = raw
    text_fields = {
        "summary": "description", "description": "description",
        "readme_excerpt": "readme", "content": "snippet", "raw_content": "page_content",
        **(text_types or {}),
    }
    results = []
    for index, record in enumerate(records):
        evidence, seen = [], set()
        # Keep each distinct text once; retain other provider fields as metadata.
        for field, text_type in text_fields.items():
            text = record.get(field)
            if text and text not in seen:
                evidence.append({"type": text_type, "text": text})
                seen.add(text)
        url = record.get("url") or ""
        resource = Resource(
            id=f"{call_id}:{index}", kind=kind,
            title=record.get("title") or record.get("name") or "", url=url,
            metadata={key: value for key, value in record.items()
                      if key not in {"title", "name", "url", *text_fields}
                      and value is not None and value != ""},
            sources=[{
                "tool": tool_name, "url": url, "evidence": evidence,
                # References address the retained response, not an invented file.
                "raw_ref": call_id if stored is not None else None,
            }],
        )
        results.append(resource.model_dump(exclude_none=True))
    return results


# --- Page enrichment ---
# enrich_url -> cached download -> extraction -> source evidence
# Underscored helpers are internal; agents call enrich_url.


def _extract_pdf(pdf_bytes: bytes, max_pages: int, max_chars: int) -> dict:
    """Extract embedded PDF text from the first pages as Markdown.

    The caller runs this helper in a subprocess to enforce its deadline.

    Args:
        pdf_bytes: Downloaded PDF contents.
        max_pages: Maximum number of pages to extract, starting at page one.
        max_chars: Maximum number of Markdown characters to return.

    Returns:
        Extracted text and a truncated flag covering both page and text limits.
    """
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=NativePdfPipeline)},
    )
    # A page range reads the prefix; max_num_pages would reject longer documents.
    conversion = converter.convert(DocumentStream(name="paper.pdf", stream=BytesIO(pdf_bytes)),
                                   page_range=(1, max_pages))
    document = conversion.document
    pages = []
    for number in sorted(document.pages):
        content = document.export_to_markdown(page_no=number).strip()
        if content:
            pages.append(f"[Page {number}]\n{content}")
    text = "\n\n".join(pages).strip()
    return {"text": text[:max_chars],
            "truncated": conversion.input.page_count > max_pages or len(text) > max_chars}


async def _check_page_url(url: str):
    """Reject unsupported URLs and non-public DNS targets before fetching.

    Args:
        url: Candidate HTTP(S) page URL.

    Raises:
        ValueError: The URL has credentials, an unsupported scheme, no host,
            or a DNS result containing a private or local address.
    """
    target = httpx.URL(url)
    if target.scheme not in {"http", "https"} or not target.host or target.userinfo:
        raise ValueError("Enrichment requires a public HTTP(S) URL without credentials.")
    # Check all resolved addresses before attempting the page request.
    addresses = await asyncio.get_running_loop().getaddrinfo(
        target.host, target.port or (443 if target.scheme == "https" else 80), type=socket.SOCK_STREAM,
    )
    if not addresses or any(not ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError("Enrichment cannot fetch private or local addresses.")


@stamina.retry(on=retry_http_error, attempts=4)
async def _download_page(url: str):
    """Download HTML or a PDF with redirect and decoded-body size checks.

    Args:
        url: Initial page URL; redirect targets receive the same checks.

    Returns:
        Final URL, response bytes, and normalized content type. The bytes are
        None when the response has an unsupported content type.

    Raises:
        ValueError: A URL check, redirect limit, or download size limit fails.
    """
    async with search_session() as client:
        for _ in range(6):
            # Redirect targets need the same checks as the original source URL.
            await _check_page_url(url)
            async with client.stream("GET", url, timeout=config.REQUEST_TIMEOUT, follow_redirects=False) as response:
                if response.is_redirect:
                    url = str(response.url.join(response.headers["location"]))
                    continue
                response.raise_for_status()
                # Avoid downloading formats that neither extractor supports.
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"text/html", "application/xhtml+xml", "application/pdf"}:
                    return str(response.url), None, content_type
                # Bound decoded body size while streaming, including compressed responses.
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > config.ENRICHMENT_MAX_BYTES:
                        raise ValueError("Page exceeds the enrichment download size limit.")
                return str(response.url), bytes(body), content_type
        raise ValueError("Too many enrichment redirects.")


async def _fetch_page(url: str) -> dict:
    """Extract page evidence under the active session's concurrency limits.

    Args:
        url: Page URL to download and extract within the active-page deadline.

    Returns:
        A status dictionary with source evidence and metadata on success,
        or details explaining a skipped or failed extraction.
    """
    started = perf_counter()
    try:
        # The active-page deadline starts after a concurrency slot becomes available.
        async with _page_limit.get(), asyncio.timeout(config.ENRICHMENT_TIMEOUT):
            final_url, body, content_type = await _download_page(url)
            if body is None:
                return {"status": "skipped", "reason": "Not an HTML page or PDF."}
            if content_type == "application/pdf":
                # A separate process enforces a hard limit on PDF parsing time.
                process = await asyncio.create_subprocess_exec(
                    sys.executable, str(Path(__file__).resolve()),
                    str(config.PDF_MAX_PAGES), str(config.PDF_MAX_CHARS),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, _ = await process.communicate(body)
                except BaseException:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                    raise
                if process.returncode:
                    raise ValueError("Docling PDF extraction failed.")
                page = json.loads(stdout)
                if not page["text"]:
                    return {"status": "skipped", "reason": "No embedded PDF text extracted."}
                reference = uuid4().hex
                raw_responses()[reference] = {"url": final_url, "extracted": page}
                source = Source(tool="docling", url=final_url, raw_ref=reference,
                                evidence=[{"type": "pdf_content", "text": page["text"]}])
                return {"status": "success", "source": source.model_dump(),
                        "metadata": {"format": "pdf", "truncated": page["truncated"]},
                        "elapsed_seconds": perf_counter() - started}
            # Run synchronous extraction off the event loop.
            extracted = await asyncio.to_thread(
                trafilatura.extract, body, url=final_url, output_format="json",
                with_metadata=True, include_comments=False, include_tables=True, include_links=True,
            )
            page = json.loads(extracted) if extracted else {}
            if not page.get("text"):
                return {"status": "skipped", "reason": "No main content extracted."}
            # Keep page metadata separate and retain the HTML behind a raw reference.
            metadata = {key: page[key] for key in (
                "title", "author", "hostname", "date", "license", "source", "categories", "tags",
            ) if page.get(key)}
            reference = uuid4().hex
            raw_responses()[reference] = {"url": final_url, "html": body.decode("utf-8", errors="replace"),
                                          "extracted": page}
            source = Source(tool="trafilatura", url=final_url, raw_ref=reference,
                            evidence=[{"type": "page_content", "text": page["text"]}])
            return {"status": "success", "source": source.model_dump(), "metadata": metadata,
                    "elapsed_seconds": perf_counter() - started}
    except Exception as exc:
        logger.warning("Page enrichment failed for %s (%s)", url, type(exc).__name__)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": perf_counter() - started}


async def enrich_url(url: str) -> dict:
    """Fetch and extract a URL once per run, sharing concurrent requests.

    Args:
        url: Page URL; fragments are ignored when looking up cached work.

    Returns:
        Cached extraction status, with source evidence on success or skip/error
        details otherwise. A standalone call creates its own session.
    """
    async with search_session():
        # URL fragments share one page download; other URL differences remain distinct.
        key = url.split("#", 1)[0]
        tasks = _page_tasks.get()
        if key not in tasks:
            tasks[key] = asyncio.create_task(_fetch_page(key))
        # Cancelling one agent must not cancel a page shared with another agent.
        return await asyncio.shield(tasks[key])


# --- Run sessions ---
# Open shared client and caches -> reuse across agents -> clean up


@asynccontextmanager
async def search_session():
    """Share HTTP, raw-response storage, and page caches within a research run.

    Nested callers reuse the current session. Only its owner cancels unfinished
    page tasks, closes the client, and restores the previous context.

    Yields:
        The active asynchronous HTTP client.
    """
    # Nested calls reuse the active session.
    client = _http_client.get()
    if client is not None:
        yield client
        return
    # The outermost call owns the client, caches, and concurrency limit.
    async with httpx.AsyncClient() as client:
        token = _http_client.set(client)
        raw_token = _raw_responses.set({})
        tasks = {}
        page_token = _page_tasks.set(tasks)
        limit_token = _page_limit.set(asyncio.Semaphore(config.ENRICHMENT_CONCURRENCY))
        try:
            yield client
        finally:
            # Cancel and await unfinished page tasks before closing the client.
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            # Restore the context, including after errors or cancellation.
            _page_limit.reset(limit_token)
            _page_tasks.reset(page_token)
            _raw_responses.reset(raw_token)
            _http_client.reset(token)


if __name__ == "__main__":
    print(json.dumps(_extract_pdf(sys.stdin.buffer.read(), *map(int, sys.argv[1:]))))

