"""Provider search tools that retain source evidence and original responses."""

import asyncio
import logging
import os
from functools import lru_cache, partial

import arxiv
import httpx
import stamina

from langchain_core.tools import ToolException, tool
from langchain_tavily import TavilySearch

import config
from utils import normalize_results, retry_http_error, search_session

logger = logging.getLogger(__name__)

# --- AI web search: Tavily ---
# Provider queries -> ranked excerpts -> normalized webpage resources


@lru_cache(maxsize=1)
def _get_tavily() -> TavilySearch:
    """Initialize and cache Tavily when a configured search first needs it."""
    # Defer credential validation until a tool call can report the failure.
    return TavilySearch(**config.AI_SEARCH_CONFIG)


@tool("tavily_search")
async def ai_search(query: str) -> list[dict]:
    """Search the web through Tavily and return normalized resources.

    Args:
        query: Focused question or search phrase for Tavily.

    Returns:
        Webpage resource dictionaries with source evidence and provider metadata.
    """
    if not os.getenv("TAVILY_API_KEY"):
        raise ToolException("Set TAVILY_API_KEY in .env.")
    response = await _get_tavily().ainvoke({"query": query})
    # The SDK may return an error payload instead of raising an exception.
    if isinstance(response, str) or response.get("error"):
        raise ToolException(response if isinstance(response, str) else str(response["error"]))
    return normalize_results("tavily_search", "webpage", response.get("results", []), response)


ai_search.description = """Search the web through Tavily using a focused question or search phrase.

Provider guidance: https://docs.tavily.com/documentation/best-practices/best-practices-search
- Keep one focused topic per query, under 1500 characters. Our basic search
  depth favors short lookups over long research instructions.
- Preserve names, relevant context, and the user's actual information need.
  Do not combine unrelated topics or return a list of subqueries.
- Domain, date, topic, and search-depth settings are separate API arguments.
  This preparation step supplies query text only; do not claim filters were set.
- Do not assume Google operators or arXiv Boolean syntax apply here.
Returns ranked URLs and relevant excerpts; excerpts are not full-page verification.
"""


# --- Grounded AI search: Gemini ---
# Search snippets and cited answer spans -> source-attributed webpage resources


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_gemini(query: str) -> list[dict]:
    """Search with Gemini and Google Search grounding; return cited web resources.

    Provider guidance: https://ai.google.dev/gemini-api/docs/google-search
    Use a focused natural-language question, preserving the topic and constraints.
    Gemini chooses its search queries; do not use provider-specific Boolean syntax.
    Returned answer spans are generated synthesis, not verbatim page content.
    Requires GEMINI_API_KEY or GOOGLE_API_KEY.

    Args:
        query: Focused research question for Google-grounded Gemini search.

    Returns:
        Webpage resources with separately labeled snippets and cited generated
        spans. Uncited generated text remains only in the raw response.
    """
    # Validate credentials before sending the request.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ToolException("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env.")
    # Enable Google Search grounding through the shared async client.
    async with search_session() as client:
        response = await client.post(
            config.GEMINI_API,
            headers={"x-goog-api-key": api_key},
            json={"model": config.GEMINI_SEARCH_MODEL, "input": query,
                  "generation_config": {"thinking_level": "low"},
                  "tools": [{"type": "google_search"}]},
            timeout=config.TOOL_TIMEOUTS["search_gemini"],
        )
    # Reject HTTP failures and incomplete interactions before parsing evidence.
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "completed":
        raise ToolException(f"Gemini search did not complete: {data.get('status', 'unknown')}")

    # Group search snippets and cited answer spans by source URL.
    records = {}
    for step in data.get("steps", []):
        if step["type"] == "google_search_result":
            for result in step.get("result", []):
                if result.get("url"):
                    record = records.setdefault(result["url"], {"url": result["url"]})
                    record.update(title=result.get("title") or result["url"], content=result.get("snippet", ""))
        elif step["type"] == "model_output":
            # Ignore non-text blocks and uncited generated claims.
            for block in step.get("content", []):
                if block["type"] != "text":
                    continue
                text = block.get("text", "").encode("utf-8")
                for citation in block.get("annotations") or []:
                    if citation["type"] != "url_citation" or not citation.get("url"):
                        continue
                    record = records.setdefault(citation["url"], {
                        "url": citation["url"], "title": citation.get("title") or citation["url"],
                    })
                    # Citation offsets are UTF-8 bytes; attach only the attributed span.
                    start, end = citation.get("start_index"), citation.get("end_index")
                    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
                        span = text[start:end].decode("utf-8", errors="replace")
                        previous = record.get("summary", "")
                        if span not in previous:
                            record["summary"] = "\n".join(filter(None, (previous, span)))
    # Normalize evidence types and retain the complete interaction for debugging.
    return normalize_results("search_gemini", "webpage", list(records.values()), data,
                             {"summary": "grounded_summary"})


# --- Web search: Serper ---


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_serper(query: str, k: int = 5) -> list[dict]:
    """Search Google through Serper; return titles, snippets, and URLs.

    Google syntax: https://support.google.com/websearch/answer/2466433
    - Use a focused search phrase or question retaining the topic and intent.
    - Double quotes require a word/phrase; site:DOMAIN restricts the site;
      -word excludes a term; before:YYYY-MM-DD and after:YYYY-MM-DD limit dates.
      Do not put spaces after operator prefixes.
    - Add restrictions only when needed for the user's request. Do not use
      arXiv field syntax or append generic research instructions.
    - Pass query text, not a URL or JSON request. This wrapper exposes query
      and k; localization and other search settings are not prepared here.
    Search snippets provide discovery evidence, not full-page verification.

    Args:
        query: Google search phrase, optionally using supported operators.
        k: Maximum number of organic results to retain.

    Returns:
        Webpage resources containing titles, snippets, and source URLs.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ToolException("Set SERPER_API_KEY in .env.")

    async with search_session() as client:
        response = await client.post(
            config.SERPER_API,
            json={"q": query},
            headers={"X-API-KEY": api_key},
            timeout=20,
        )
    response.raise_for_status()
    data = response.json()
    # Normalize organic results as snippets, preserving the full response separately.
    return normalize_results("search_serper", "webpage", [
        {
            "title": result.get("title", ""),
            "summary": result.get("snippet", ""),
            "url": result.get("link", ""),
        }
        for result in data.get("organic", [])[:k]
    ], data, {"summary": "snippet"})


# --- Paper text search: arXiv ---


@tool
async def search_arxiv(query: str) -> list[dict]:
    """Search academic papers using arXiv's fielded/Boolean query syntax.

    Official syntax: https://info.arxiv.org/help/api/user-manual.html#51-details-of-query-construction
    - Fields: all: (all searchable metadata), ti: (title), abs: (abstract),
      au: (author), cat: (category), co: (comment), jr: (journal), rn: (report).
    - Uppercase AND requires both terms; OR allows alternatives; ANDNOT
      excludes matches. Use parentheses to group expressions.
    - Double quotes match phrases, e.g. ti:"fashion retrieval".
    - Submission dates: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM], in GMT.
    - Pass plain query text; the SDK handles URL encoding.
    - ID/version lookups require id_list rather than id:; this tool does
      not expose id_list.

    Prefer concise search terms over research instructions. Add exclusions,
    phrase restrictions, or author/category/date filters only when needed
    for the user's request. Example: all:fashion AND
    (all:embedding OR all:embeddings). Exclusion example:
    all:fashion ANDNOT (ti:medical OR ti:clinical).

    Args:
        query: Plain arXiv query using its fielded or Boolean syntax.

    Returns:
        Paper resources with titles, abstracts, publication dates, and URLs.
    """
    def fetch():
        """Read paper records and raw SDK metadata inside the worker thread."""
        search = arxiv.Search(query=query, max_results=config.ARXIV_MAX_RESULTS)
        client = arxiv.Client()
        # The SDK has no public timeout setting; bound its underlying requests.
        with client._session as session:
            session.request = partial(session.request, timeout=config.ARXIV_REQUEST_TIMEOUT)
            papers = list(client.results(search))
        return [
            {
                "title": p.title,
                "url": p.entry_id,
                "published": p.published.date().isoformat(),
                "summary": p.summary,
            }
            for p in papers
        ], {"results": [vars(p) for p in papers]}

    # The arXiv SDK is synchronous, including iteration over its results.
    records, raw = await asyncio.to_thread(fetch)
    return normalize_results("search_arxiv", "paper", records, raw, {"summary": "abstract"})


# --- Repository search: GitHub ---
# Repository matches -> concurrent README fetches -> bounded evidence excerpts


@stamina.retry(on=retry_http_error, attempts=4)
async def _fetch_github_readme(repo_name: str, headers: dict) -> str:
    """Fetch a repository's full README before the caller limits its excerpt.

    Args:
        repo_name: Repository identifier in owner/name form.
        headers: GitHub request headers, including authentication when available.

    Returns:
        The complete README text returned by GitHub.
    """
    async with search_session() as client:
        response = await client.get(
            f"https://api.github.com/repos/{repo_name}/readme",
            headers={**headers, "Accept": "application/vnd.github.raw+json"},
            timeout=config.REQUEST_TIMEOUT,
            follow_redirects=True,
        )
    response.raise_for_status()
    return response.text


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_github(query: str) -> list[dict]:
    """Find GitHub repositories by relevance and return README excerpts.

    Official syntax: https://docs.github.com/en/search-github/searching-on-github/searching-for-repositories
    - Use concise keywords; double quotes match multi-word phrases.
    - Scope with in:name,description,topics,readme (the tool's default),
      user:OWNER, org:ORG, or repo:OWNER/NAME.
    - Filter with language:python, topic:machine-learning, license:mit,
      archived:true/false, or is:public/private (subject to access).
    - Numeric qualifiers such as stars:, forks:, and size: (KB) accept
      comparisons (>, >=, <, <=) and inclusive ranges (10..100).
    - created: and pushed: accept YYYY-MM-DD dates, comparisons, and ranges.
    - NOT excludes keywords; -QUALIFIER excludes qualifier matches.
      fork:true includes forks; fork:only searches only forks.
    Add restrictions only when needed for the user's request. Do not use
    arXiv syntax or code-search-only qualifiers such as path: or symbol:.
    Example: "fashion retrieval" language:python NOT tutorial.
    This searches repositories, not source files. README excerpts may be None.

    Args:
        query: GitHub repository search terms and supported qualifiers.

    Returns:
        Repository resources with metadata and available README excerpts.
    """
    # Default search scope.
    if not any(word.lower().startswith("in:") for word in query.split()):
        query = f"{query} in:name,description,topics,readme"

    # Optional authentication.
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def build_result(repo):
        """Retain a full README in the provider record and return its excerpt.

        Args:
            repo: Repository search record to enrich in place.

        Returns:
            Common repository fields, including an optional README excerpt.
        """
        # Keep the repository result even when its README is unavailable.
        try:
            readme = await _fetch_github_readme(repo["full_name"], headers)
        except httpx.HTTPError as exc:
            logger.warning(
                "README unavailable for %s (%s)",
                repo["full_name"], type(exc).__name__,
            )
            readme = None

        # Retain the full README; only the LLM excerpt is bounded.
        repo["readme_text"] = readme
        return {
            "name": repo["full_name"],
            "url": repo["html_url"],
            "description": repo["description"],
            "stars": repo["stargazers_count"],
            "readme_excerpt": readme[:config.GITHUB_README_MAX_CHARS] if readme is not None else None,
        }

    async with search_session() as client:
        response = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": config.GITHUB_MAX_RESULTS},
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        response.raise_for_status()
        data = response.json()
        # Fetch independent READMEs concurrently through the shared client.
        records = await asyncio.gather(*(build_result(repo) for repo in data["items"]))
    return normalize_results("search_github", "repository", records, data)


# --- File-content search: GitHub ---
# Indexed code matches -> exact blob contents -> bounded code evidence


@stamina.retry(on=retry_http_error, attempts=4)
async def _fetch_github_code(repository: str, blob_sha: str, headers: dict) -> str:
    """Fetch the exact file version identified by a GitHub search match.

    Args:
        repository: Repository identifier in owner/name form.
        blob_sha: Indexed Git blob SHA to retrieve.
        headers: GitHub request headers, including authentication.

    Returns:
        The complete file text returned by GitHub.
    """
    # Fetch the indexed blob rather than a potentially changed default-branch file.
    async with search_session() as client:
        response = await client.get(
            f"https://api.github.com/repos/{repository}/git/blobs/{blob_sha}",
            headers={**headers, "Accept": "application/vnd.github.raw+json"},
            timeout=config.REQUEST_TIMEOUT,
        )
    response.raise_for_status()
    return response.text


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_github_code(query: str) -> list[dict]:
    """Search GitHub file contents and retrieve code excerpts from matching files.

    Official syntax: https://docs.github.com/en/rest/search/search#search-code
    Use concise code terms or quoted phrases, with repo:OWNER/NAME, org:OWNER,
    language:python, path:src, filename:train.py, or extension:py as needed.
    Use REST code-search syntax, not regex, symbol: queries, or repository-only
    filters such as stars: and license:. Pass plain, unencoded query text.
    Scope to a repository only when the user supplies it or evidence identifies it.
    Requires GITHUB_TOKEN. Returns paths, indexed snippets, blob SHAs, and code
    excerpts; file-fetch failures retain search snippets. Retrieved code is not executed.

    Args:
        query: Plain GitHub REST code-search terms and supported qualifiers.

    Returns:
        Repository resources representing matched files, with code excerpts,
        search snippets, and retrieval or truncation metadata.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ToolException("Set GITHUB_TOKEN in .env for GitHub code search.")
    headers = {"Accept": "application/vnd.github.text-match+json", "Authorization": f"Bearer {token}"}

    async def build_result(item):
        """Retain full file text and build a bounded evidence record.

        Args:
            item: Code-search match to enrich in place with retrieved file text.

        Returns:
            File identity, indexed snippets, code excerpt, and retrieval status.
        """
        repository = item["repository"]["full_name"]
        code, error = None, None
        try:
            code = await _fetch_github_code(repository, item["sha"], headers)
        except httpx.HTTPError as exc:
            # Keep the indexed match when the full file cannot be retrieved.
            error = f"Code unavailable ({type(exc).__name__})"
            logger.warning("Code unavailable for %s:%s (%s)", repository, item["path"], type(exc).__name__)
        item["code_text"] = code
        return {
            "title": f"{repository}:{item['path']}", "url": item["html_url"],
            "repository": repository, "path": item["path"], "blob_sha": item["sha"],
            "summary": "\n".join(match["fragment"] for match in item.get("text_matches", [])),
            "content": code[:config.GITHUB_CODE_MAX_CHARS] if code is not None else None,
            "code_retrieved": code is not None,
            "code_truncated": len(code) > config.GITHUB_CODE_MAX_CHARS if code is not None else False,
            "code_error": error,
        }

    async with search_session() as client:
        response = await client.get(
            "https://api.github.com/search/code", params={"q": query, "per_page": config.GITHUB_MAX_RESULTS},
            headers=headers, timeout=config.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        # Fetch matching files concurrently, retaining full code in raw responses.
        records = await asyncio.gather(*(build_result(item) for item in data["items"][:config.GITHUB_MAX_RESULTS]))
    return normalize_results("search_github_code", "repository", records, data,
                             {"summary": "code_snippet", "content": "code"})


# --- Semantic dataset and model search: Hugging Face ---
# Natural-language queries -> provider records -> common resources


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_huggingface(query: str, k: int = 5) -> list[dict]:
    """Find Hugging Face datasets by meaning through the community Hub Search API.

    Service reference: https://davanstrien-hub-search-api.hf.space/docs
    - Describe the desired data, modality, domain, annotations, and intended
      task in natural language. Retain useful context and explicit constraints.
    - Use semantic descriptions, not Boolean expressions or site: qualifiers.
      The service adds its embedding-query prefix; do not add one yourself.
    - This wrapper exposes query and k only. Requirements in text are not
      guaranteed metadata filters; verify properties in the returned sources.
    Returns dataset names, provider summaries, and URLs, not complete cards.

    Args:
        query: Natural-language description of the desired dataset.
        k: Number of results requested from the community search service.

    Returns:
        Dataset resources with provider summaries and Hugging Face URLs.
    """
    async with search_session() as client:
        response = await client.get(
            f"{config.HF_SEARCH_API}/search/datasets",
            params={"query": query, "k": k},
            timeout=20,
        )
    response.raise_for_status()
    data = response.json()
    return normalize_results("search_huggingface", "dataset", [
        {
            "name": dataset["dataset_id"],
            "summary": dataset.get("summary", ""),
            "url": f"https://huggingface.co/datasets/{dataset['dataset_id']}",
        }
        for dataset in data.get("results", [])
    ], data)


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_models(
    query: str,
    k: int = 5,
    min_params_b: float | None = None,
    max_params_b: float | None = None,
    task: str | None = None,
    license: str | None = None,
) -> list[dict]:
    """Find Hugging Face models by meaning through the community Hub Search API.

    Service reference: https://davanstrien-hub-search-api.hf.space/docs
    - Describe the task, input/output modalities, domain, and desired model
      capabilities in natural language. Preserve useful constraints and names.
    - Do not use Boolean/site: syntax or embedding-query prefixes; the service
      handles query encoding and embedding prompts.
    - Size, task, and license filters are separate arguments. Size bounds are
      in billions (min_params_b=1, max_params_b=7). Query preparation produces
      text only; do not claim an API filter was applied from query wording.
    Returns names, provider summaries, parameter counts, licenses, and URLs.

    Args:
        query: Natural-language description of the desired model.
        k: Number of results requested from the community search service.
        min_params_b: Optional minimum parameter count in billions.
        max_params_b: Optional maximum parameter count in billions.
        task: Optional provider task filter.
        license: Optional provider license filter.

    Returns:
        Model resources with available summaries, parameter counts, and licenses.
    """
    # Convert size bounds to provider units; filters remain separate from query text.
    params = {"query": query, "k": k}
    if min_params_b is not None:
        params["min_param_count"] = int(min_params_b * 1e9)
    if max_params_b is not None:
        params["max_param_count"] = int(max_params_b * 1e9)
    if task:
        params["task"] = task
    if license:
        params["license"] = license
    async with search_session() as client:
        response = await client.get(
            f"{config.HF_SEARCH_API}/search/models", params=params, timeout=20,
        )
    response.raise_for_status()
    data = response.json()
    return normalize_results("search_models", "model", [
        {
            "name": model["model_id"],
            "summary": model.get("summary", ""),
            "params": model.get("param_count"),
            "license": model.get("license"),
            "url": f"https://huggingface.co/{model['model_id']}",
        }
        for model in data.get("results", [])
    ], data)


# --- Dataset and model text search: Kaggle ---
# Keyword queries -> authenticated searches -> common resources


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_kaggle(keywords: str, k: int = 5) -> list[dict]:
    """Search Kaggle datasets by keyword; return summaries, URLs, and metadata.

    Official search options: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md
    - Pass a concise topic phrase, e.g. fashion embeddings or garment keypoints.
      Preserve relevant dataset names; omit research instructions.
    - Do not include Boolean operators, field prefixes, or CLI flags:
      Kaggle documents a search term, not an advanced query grammar.
    - License, tags, owner, file type, and size are separate filters; this
      wrapper does not expose them. Do not encode filters into the keywords.
    Results are sorted by votes and limited to k. Available metadata includes
    description, license, size_bytes, and last_updated. Requires KAGGLE_API_TOKEN.

    Args:
        keywords: Concise dataset search terms for Kaggle.
        k: Maximum number of dataset results to retain.

    Returns:
        Dataset resources with summaries, URLs, and available provider metadata.
    """
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise ToolException("Set KAGGLE_API_TOKEN in .env.")

    async with search_session() as client:
        response = await client.get(
            config.KAGGLE_API,
            params={"search": keywords, "sortBy": "votes"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
    response.raise_for_status()
    # Map optional provider fields without inventing missing metadata.
    metadata_fields = {
        "description": "description",
        "license": "licenseName",
        "size_bytes": "totalBytes",
        "last_updated": "lastUpdated",
    }
    data = response.json()
    return normalize_results("search_kaggle", "dataset", [
        {
            "name": dataset.get("ref", ""),
            "summary": dataset.get("subtitle") or dataset.get("title", ""),
            "url": f"https://www.kaggle.com/datasets/{dataset.get('ref', '')}",
            **{
                field: dataset[source]
                for field, source in metadata_fields.items()
                if dataset.get(source) not in (None, "")
            },
        }
        for dataset in data[:k]
    ], data)


@tool
@stamina.retry(on=retry_http_error, attempts=4)
async def search_kaggle_models(keywords: str, k: int = 5) -> list[dict]:
    """Search Kaggle models by keyword; return names, summaries, and URLs.

    Official search options: https://github.com/Kaggle/kaggle-cli/blob/main/docs/models.md
    - Pass concise task, model, or architecture terms, e.g. fashion embeddings
      or bert. Preserve relevant model names; omit research instructions.
    - Do not include Boolean operators, field prefixes, or CLI flags:
      Kaggle documents keyword search, not an advanced query grammar.
    - Owner filtering and sorting are separate options; this wrapper exposes
      only keywords and k. Do not claim that additional filters were applied.
    Requires KAGGLE_API_TOKEN.

    Args:
        keywords: Concise model, architecture, or task terms for Kaggle.
        k: Maximum number of model results to request and retain.

    Returns:
        Model resources with names, summaries, and source URLs.
    """
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        raise ToolException("Set KAGGLE_API_TOKEN in .env.")

    async with search_session() as client:
        response = await client.get(
            config.KAGGLE_MODELS_API,
            params={"search": keywords, "pageSize": k},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
    response.raise_for_status()
    data = response.json()
    return normalize_results("search_kaggle_models", "model", [
        {
            "name": model.get("ref", ""),
            "summary": model.get("subtitle") or model.get("title", ""),
            "url": model.get("url") or f"https://www.kaggle.com/models/{model.get('ref', '')}",
        }
        for model in data.get("models", [])[:k]
    ], data)


# --- Semantic paper search: OpenAlex ---
# Natural-language queries -> reconstructed abstracts -> paper resources


def _openalex_abstract(inverted: dict | None) -> str:
    """Reconstruct an OpenAlex abstract from word positions.

    Args:
        inverted: Mapping of words to their positions, or None for no abstract.

    Returns:
        Words in their original order, or an empty string when unavailable.
    """
    words = {position: word for word, positions in (inverted or {}).items() for position in positions}
    return " ".join(words[position] for position in sorted(words))


@tool
@stamina.retry(on=retry_http_error, attempts=4, wait_initial=1)
async def search_openalex(
    query: str, k: int = 5, year_from: int | None = None,
) -> list[dict]:
    """Search OpenAlex papers by meaning using a natural-language query.

    Provider guidance: https://help.openalex.org/api/semantic-search/
    - Describe the research problem, methods, and context in natural language.
      Useful abstract/paragraph context improves matching; retain it rather
      than reducing everything to keywords. Only the first 2000 characters
      are used for matching.
    - This uses search.semantic, not lexical search: do not add Boolean
      operators, field qualifiers, URLs, or URL encoding to the query.
    - year_from is a separate inclusive year filter; this preparation step
      supplies text only. Other search modes cannot be combined with semantic
      search. Results may use different terminology from the query.
    Returns titles, years, full abstracts when available, citation counts, and URLs.

    Args:
        query: Natural-language description of the research topic.
        k: Number of paper results requested from OpenAlex.
        year_from: Optional inclusive starting publication year.

    Returns:
        Paper resources with reconstructed abstracts and available metadata.
    """
    api_key = os.getenv("OPENALEX_API_KEY")
    params = {
        "search.semantic": query,
        "per_page": k,
        "select": "id,display_name,publication_year,cited_by_count,doi,abstract_inverted_index",
    }
    # Express an inclusive starting year using the provider's strict comparison.
    if year_from is not None:
        params["filter"] = f"publication_year:>{year_from - 1}"
    async with search_session() as client:
        response = await client.get(
            config.OPENALEX_API,
            params=params,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=30,
        )
    response.raise_for_status()
    data = response.json()
    return normalize_results("search_openalex", "paper", [
        {
            "title": paper.get("display_name") or "",
            "year": paper.get("publication_year"),
            "summary": _openalex_abstract(paper.get("abstract_inverted_index")),
            "citations": paper.get("cited_by_count"),
            "url": paper.get("doi") or paper.get("id", ""),
        }
        for paper in data.get("results", [])
    ], data, {"summary": "abstract"})


# --- Tool registration ---
# Credential errors become tool messages; the shared agent handles other failures.

ai_search.handle_tool_error = True
search_kaggle.handle_tool_error = True
search_kaggle_models.handle_tool_error = True
search_serper.handle_tool_error = True
search_gemini.handle_tool_error = True
search_github_code.handle_tool_error = True

# Default tools for the shared research agent.
tools = [ai_search, search_gemini, search_serper, search_arxiv, search_github, search_github_code, search_huggingface, search_models, search_kaggle, search_kaggle_models, search_openalex]
