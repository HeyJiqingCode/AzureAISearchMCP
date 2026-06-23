import asyncio
import io
import json
import logging
import os
import re
import sys
from argparse import ArgumentParser
from typing import Any, Dict, Final, List, Optional
from urllib.parse import quote, unquote

import aiohttp
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

_utf8_stderr = None


def configure_utf8_logging() -> None:
    global _utf8_stderr

    if _utf8_stderr is None:
        _utf8_stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(_utf8_stderr)
    formatter = logging.Formatter(fmt="[%(levelname)-8s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def _comma_split(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


configure_utf8_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP("Azure AI Search MCP Server")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
DEFAULT_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY")
DEFAULT_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
HTTP_TIMEOUT_SECONDS: Final[int] = int(os.getenv("AZURE_SEARCH_TIMEOUT", "30"))
AGENTIC_HTTP_TIMEOUT_SECONDS: Final[int] = int(
    os.getenv("AZURE_SEARCH_AGENTIC_TIMEOUT", os.getenv("AZURE_SEARCH_TIMEOUT", "90"))
)
AGENTIC_TIMEOUT_BUFFER_SECONDS: Final[int] = int(os.getenv("AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER", "30"))
AGENTIC_API_VERSION: Final[str] = "2025-11-01-preview"


def _resolve_endpoint(endpoint: Optional[str] = None) -> str:
    resolved = endpoint or AZURE_SEARCH_ENDPOINT
    if not resolved:
        raise RuntimeError(
            "Azure Search endpoint is not configured. Set AZURE_SEARCH_ENDPOINT or pass endpoint explicitly."
        )
    return resolved.rstrip("/")


def _resolve_key(explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    if DEFAULT_QUERY_KEY:
        return DEFAULT_QUERY_KEY
    raise RuntimeError("Azure Search API key is not configured. Provide api_key or set AZURE_SEARCH_QUERY_KEY.")


def _resolve_admin_key(explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    if DEFAULT_ADMIN_KEY:
        return DEFAULT_ADMIN_KEY
    raise RuntimeError(
        "Agentic retrieval requires an admin key. Provide api_key or set AZURE_SEARCH_ADMIN_KEY."
    )


async def _maybe_await(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return await result
    return result


def _build_messages_from_query(query: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                }
            ],
        }
    ]


def _build_agentic_timeout_budget(max_runtime_seconds: Optional[int]) -> int:
    """计算 Agentic Retrieval 的客户端超时预算。

    参数:
        max_runtime_seconds: 请求体中的服务端最大运行时长；为 None 时表示未显式限制。

    返回:
        int: 供 aiohttp 使用的总超时秒数，包含服务端运行时间和额外缓冲。
    """
    requested_runtime_seconds: int = max_runtime_seconds or 0
    buffered_runtime_seconds: int = requested_runtime_seconds + AGENTIC_TIMEOUT_BUFFER_SECONDS
    return max(AGENTIC_HTTP_TIMEOUT_SECONDS, buffered_runtime_seconds)


def _normalize_document(document: Dict[str, Any]) -> Dict[str, Any]:
    def _default(obj: Any) -> Any:
        # Handle datetime objects
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except Exception:  # pragma: no cover - fallback path
                return str(obj)
        # Handle objects with as_dict method
        if hasattr(obj, "as_dict"):
            try:
                result = obj.as_dict()
                # Recursively normalize the dict result
                if isinstance(result, dict):
                    return {k: _default(v) for k, v in result.items()}
                return result
            except Exception:  # pragma: no cover - fallback path
                return str(obj)
        # Handle objects with __dict__
        if hasattr(obj, "__dict__") and obj.__dict__:
            try:
                return {key: _default(value) for key, value in obj.__dict__.items() if not key.startswith("_")}
            except Exception:  # pragma: no cover
                return str(obj)
        # Handle basic types
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        # Handle lists
        if isinstance(obj, list):
            return [_default(item) for item in obj]
        # Handle dicts
        if isinstance(obj, dict):
            return {k: _default(v) for k, v in obj.items()}
        # Fallback: convert to string
        return str(obj)

    serialized: Dict[str, Any] = {}
    for key, value in document.items():
        serialized[key] = _default(value)
    return serialized


def _serialize_highlights(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _serialize_facet_entry(item: Any) -> Dict[str, Any]:
    if hasattr(item, "value") or hasattr(item, "count"):
        return {
            "value": getattr(item, "value", None),
            "count": getattr(item, "count", None),
        }
    if isinstance(item, dict):
        return {
            "value": item.get("value"),
            "count": item.get("count"),
        }
    return {"value": str(item), "count": None}


def _serialize_facets(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        facets: Dict[str, Any] = {}
        for facet_key, facet_values in raw.items():
            if isinstance(facet_values, list):
                facets[facet_key] = [_serialize_facet_entry(item) for item in facet_values]
            else:
                facets[facet_key] = str(facet_values)
        return facets
    return str(raw) if raw else None


async def _collect_results(result_iterator: Any) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    async for item in result_iterator:  # each item is SearchResult (Mapping)
        items.append(_normalize_document(dict(item)))

    count = None
    answers = None
    facets = None
    captions = None

    for attr, target in (("get_count", "count"), ("get_answers", "answers"), ("get_facets", "facets"), ("get_captions", "captions")):
        if hasattr(result_iterator, attr):
            try:
                raw = await _maybe_await(getattr(result_iterator, attr)())
            except Exception:  # pragma: no cover - defensive
                continue
            if not raw:
                continue
            if target == "count":
                count = raw
            elif target == "answers":
                answers = []
                for answer in raw:
                    answers.append(
                        {
                            "key": str(getattr(answer, "key", "")) if getattr(answer, "key", None) is not None else None,
                            "text": str(getattr(answer, "text", "")) if getattr(answer, "text", None) is not None else None,
                            "score": float(getattr(answer, "score", 0.0)) if getattr(answer, "score", None) is not None else None,
                            "highlights": _serialize_highlights(getattr(answer, "highlights", None)),
                        }
                    )
            elif target == "facets":
                facets = _serialize_facets(raw)
            elif target == "captions":
                captions = []
                for caption in raw:
                    captions.append(
                        {
                            "text": str(getattr(caption, "text", "")) if getattr(caption, "text", None) is not None else None,
                            "highlights": _serialize_highlights(getattr(caption, "highlights", None)),
                        }
                    )

    continuation_token = None
    if hasattr(result_iterator, "get_continuation_token"):
        try:
            continuation_token = result_iterator.get_continuation_token()
        except Exception:  # pragma: no cover - defensive
            continuation_token = None

    # Build response dict, only include non-None values
    response = {"documents": items}

    if count is not None:
        response["count"] = count
    if answers is not None:
        response["answers"] = answers
    if facets is not None:
        response["facets"] = facets
    if captions is not None:
        response["captions"] = captions
    if continuation_token is not None:
        response["continuation_token"] = continuation_token

    return response


async def _create_search_client(
    *,
    endpoint: str,
    index_name: str,
    credential: AzureKeyCredential,
) -> SearchClient:
    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=credential,
    )
    return client


async def _execute_search(
    *,
    endpoint: str,
    key: str,
    index_name: str,
    search_text: Optional[str],
    search_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    credential = AzureKeyCredential(key)
    client = await _create_search_client(endpoint=endpoint, index_name=index_name, credential=credential)
    async with client:
        result_pager = await client.search(
            search_text=search_text,
            **search_kwargs,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        payload = await _collect_results(result_pager)
    return payload


def _build_vector_query(
    *,
    vector_text: str,
    vector_fields: Optional[str],
    k: int,
    exhaustive: bool,
    weight: Optional[float] = None,
) -> List[VectorizableTextQuery]:
    if vector_fields is None:
        raise ValueError("vector_fields must be provided for vector-enabled search.")
    vector_query = VectorizableTextQuery(
        text=vector_text,
        fields=vector_fields,
        k_nearest_neighbors=k,
        exhaustive=exhaustive,
    )
    if weight is not None:
        vector_query.weight = weight
    return [vector_query]

def _extract_page_numbers_from_documents(documents: List[Dict[str, Any]]) -> List[int]:
    """
    Extract unique page numbers from search result documents.

    Args:
        documents: List of document results from search

    Returns:
        Sorted list of unique page numbers
    """
    page_numbers = set()
    for doc in documents:
        # Try different possible field names for page number
        page_from = doc.get("page_from") or doc.get("pageNumberFrom")
        if page_from is not None:
            try:
                page_numbers.add(int(page_from))
            except (ValueError, TypeError):
                pass
    return sorted(list(page_numbers))


def _format_agentic_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format Agentic Retrieval API response for better readability.

    Transforms:
    - [ref_id:0] -> <sup>1</sup> (0-based to 1-based)
    - Web references -> "1. [Web] [title](url)"
    - KB references -> "1. [KnowledgeBase: source-name] [rerankerScore: X.XX] Title: XXX"

    Args:
        raw_response: Raw API response from Azure AI Search

    Returns:
        Formatted response with answer and references
    """
    result: Dict[str, Any] = {
        "answer": "",
        "references": []
    }

    # Extract answer text
    try:
        answer_text = raw_response["response"][0]["content"][0]["text"]
    except (KeyError, IndexError):
        answer_text = ""

    # Convert ref_id to superscript (0-based -> 1-based)
    def replace_ref(match):
        ref_id = int(match.group(1))
        superscript_num = ref_id + 1
        return f"<sup>{superscript_num}</sup>"

    formatted_answer = re.sub(r'\[ref_id:(\d+)\]', replace_ref, answer_text)
    result["answer"] = formatted_answer

    # Build activity ID to knowledge source name mapping
    activity_map = {}
    activities = raw_response.get("activity", [])
    for activity in activities:
        activity_id = activity.get("id")
        knowledge_source_name = activity.get("knowledgeSourceName")
        if activity_id is not None and knowledge_source_name:
            activity_map[activity_id] = knowledge_source_name

    # Format references
    references = raw_response.get("references", [])
    formatted_refs = []

    for ref in references:
        ref_type = ref.get("type", "")
        ref_id = ref.get("id", "")
        activity_source = ref.get("activitySource")

        # Convert ID from 0-based to 1-based
        try:
            display_id = int(ref_id) + 1
        except (ValueError, TypeError):
            display_id = ref_id

        # Get knowledge source name from activity
        knowledge_source_name = activity_map.get(activity_source, "Unknown")

        if ref_type == "web":
            # Web type: N. [Web] [title](url)
            title = ref.get("title", "Untitled")
            url = ref.get("url", "")
            formatted_refs.append(f"{display_id}. [Web] [{title}]({url})")

        elif ref_type in ["searchIndex", "remoteSharePoint"]:
            # Knowledge Base type: N. [KnowledgeBase: source-name] [rerankerScore: X.XX] Title: XXX
            title = ref.get("title", "Untitled")
            reranker_score = ref.get("rerankerScore", 0.0)
            formatted_refs.append(
                f"{display_id}. [KnowledgeBase: {knowledge_source_name}] [rerankerScore: {reranker_score:.2f}] Title: {title}"
            )
        else:
            # Unknown type, use generic format
            formatted_refs.append(f"{display_id}. [Unknown Type: {ref_type}]")

    # Sort references by display ID (numeric order)
    formatted_refs.sort(key=lambda x: int(x.split('.')[0]))

    result["references"] = formatted_refs

    return result

@mcp.tool(
    name="simple_search",
    description="Keyword (BM25) search over an index using simple query syntax, with optional filters and field selection.",
)
async def simple_search(
    index_name: str,
    query: str,
    top: int = 5,
    skip: int = 0,
    search_fields: str = "",
    select: str = "",
    filter: str = "",
    search_mode: str = "any",
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Run a standard keyword search against Azure AI Search.

    Parameters
    ----------
    index_name: str
        Target index name.
    query: str
        Simple query syntax string (terms, phrases, Boolean operators).
    top: int, optional
        Maximum number of results to return (default 5).
    skip: int, optional
        Number of results to skip for paging.
    search_fields: Optional[str]
        Comma-separated searchable fields to scope the query.
    select: Optional[str]
        Comma-separated retrievable fields in the response.
    filter: Optional[str]
        OData filter expression applied before keyword search.
    search_mode: str, optional
        "any" (default) or "all" to control precision/recall.
    api_key / endpoint: Optional[str]
        Override default query key or endpoint for this call.

    Returns
    -------
    dict
        Response containing `documents`, `count`, optional `facets`, and `continuation_token`.
    """
    # Convert empty strings to None for optional parameters
    search_fields = None if search_fields == "" else search_fields
    select = None if select == "" else select
    filter = None if filter == "" else filter
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_key(api_key)
    search_kwargs: Dict[str, Any] = {
        "top": top,
        "skip": skip,
        "include_total_count": True,
        "search_mode": search_mode,
    }
    if filter:
        search_kwargs["filter"] = filter
    fields = _comma_split(search_fields)
    if fields:
        search_kwargs["search_fields"] = fields
    selected = _comma_split(select)
    if selected:
        search_kwargs["select"] = selected

    return await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=query,
        search_kwargs=search_kwargs,
    )


@mcp.tool(
    name="semantic_search",
    description="Semantic reranked search returning optional captions and answers when the index has semantic configuration enabled.",
)
async def semantic_search(
    index_name: str,
    query: str,
    semantic_configuration: str,
    top: int = 5,
    skip: int = 0,
    select: str = "",
    filter: str = "",
    api_key: str = "",
    endpoint: str = "",
    query_caption: str = "extractive",
    query_answer: str = "",
    query_answer_count: int = 0,
    query_answer_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Execute semantic ranking against a configured index.

    Parameters
    ----------
    index_name: str
        Target index name.
    query: str
        Natural-language question or prompt for semantic ranking.
    semantic_configuration: str
        Name of the semantic configuration defined on the index.
    top / skip: int, optional
        Pagination controls (top defaults to 5).
    select / filter: Optional[str]
        Shape the fields returned and pre-filter documents.
    api_key / endpoint: Optional[str]
        Override default connection information.
    query_caption / query_answer: Optional[str]
        Semantic caption and answer modes (`"extractive"`, `"summary"`, etc.).
    query_answer_count / query_answer_threshold: Optional
        Controls for the number of answers and confidence threshold.

    Returns
    -------
    dict
        Response containing `documents`, `count`, optional `answers`, `captions`, and continuation metadata.
    """
    # Convert empty strings and sentinel values to None
    select = None if select == "" else select
    filter = None if filter == "" else filter
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint
    # query_caption has default "extractive", don't convert to None
    query_answer = None if query_answer == "" else query_answer
    query_answer_count = None if query_answer_count == 0 else query_answer_count
    query_answer_threshold = None if query_answer_threshold == 0.0 else query_answer_threshold

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_key(api_key)

    search_kwargs: Dict[str, Any] = {
        "query_type": "semantic",
        "semantic_configuration_name": semantic_configuration,
        "semantic_query": query,
        "top": top,
        "skip": skip,
        "include_total_count": True,
    }
    if select:
        search_kwargs["select"] = _comma_split(select)
    if filter:
        search_kwargs["filter"] = filter
    if query_caption:
        search_kwargs["query_caption"] = query_caption
    if query_answer:
        search_kwargs["query_answer"] = query_answer
        if query_answer_count is not None:
            search_kwargs["query_answer_count"] = query_answer_count
        if query_answer_threshold is not None:
            search_kwargs["query_answer_threshold"] = query_answer_threshold

    return await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=query,
        search_kwargs=search_kwargs,
    )


@mcp.tool(
    name="vector_search",
    description="Vector-only similarity search using integrated vectorization (text-to-embedding) over specified vector fields.",
)
async def vector_search(
    index_name: str,
    vector_fields: str,
    vector_text: str,
    k: int = 10,
    exhaustive: bool = False,
    weight: float = 0.0,
    select: str = "",
    filter: str = "",
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Perform pure vector similarity search.

    Parameters
    ----------
    index_name: str
        Target index name.
    vector_fields: str
        Comma-separated vector field names participating in search.
    vector_text: str
        Raw query text to be vectorized using the index's configured vectorizer.
    k: int, optional
        Number of nearest neighbors to retrieve (default 10).
    exhaustive: bool, optional
        Use exhaustive KNN instead of ANN when True.
    weight: Optional[float]
        Weight assigned to the vector query (relevant when mixing multiple vectors).
    select / filter: Optional[str]
        Restrict fields returned or apply filters before vector scoring.
    api_key / endpoint: Optional[str]
        Override default connection information.

    Returns
    -------
    dict
        Response containing `documents`, `count`, and `continuation_token`.
    """
    # Convert empty strings and sentinel values to None
    weight = None if weight == 0.0 else weight
    select = None if select == "" else select
    filter = None if filter == "" else filter
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_key(api_key)

    vector_queries = _build_vector_query(
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=exhaustive,
        weight=weight,
    )

    search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "top": k,
        "include_total_count": True,
    }
    if select:
        search_kwargs["select"] = _comma_split(select)
    if filter:
        search_kwargs["filter"] = filter

    return await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=None,
        search_kwargs=search_kwargs,
    )


@mcp.tool(
    name="hybrid_search",
    description="Hybrid (keyword + vector) search that fuses BM25 and vector similarity results using Reciprocal Rank Fusion.",
)
async def hybrid_search(
    index_name: str,
    query: str,
    vector_fields: str,
    vector_text: str,
    k: int = 10,
    top: int = 10,
    exhaustive: bool = False,
    weight: float = 0.0,
    select: str = "",
    filter: str = "",
    search_fields: str = "",
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Combine lexical and vector retrieval in a single request.

    Parameters
    ----------
    index_name: str
        Target index name.
    query: str
        Keyword query (simple syntax) for the lexical component.
    vector_fields: str
        Comma-separated vector fields used for similarity search.
    vector_text: str
        Raw query text to be vectorized.
    k / top: int, optional
        Vector candidate count (k) and final result count (top).
    exhaustive / weight: optional
        Control vector search exhaustive mode and weighting.
    select / filter / search_fields: Optional[str]
        Customize returned fields, filters, or lexical scope.
    api_key / endpoint: Optional[str]
        Override default connection information.

    Returns
    -------
    dict
        Response containing merged `documents`, `count`, and continuation metadata.
    """
    # Convert empty strings and sentinel values to None
    weight = None if weight == 0.0 else weight
    select = None if select == "" else select
    filter = None if filter == "" else filter
    search_fields = None if search_fields == "" else search_fields
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_key(api_key)

    vector_queries = _build_vector_query(
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=exhaustive,
        weight=weight,
    )

    search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "top": top,
        "include_total_count": True,
    }
    if select:
        search_kwargs["select"] = _comma_split(select)
    if filter:
        search_kwargs["filter"] = filter
    if search_fields:
        search_kwargs["search_fields"] = _comma_split(search_fields)

    return await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=query,
        search_kwargs=search_kwargs,
    )


@mcp.tool(
    name="semantic_hybrid_search",
    description="Hybrid (keyword + vector) search with semantic reranking, captions, and answers when configured.",
)
async def semantic_hybrid_search(
    index_name: str,
    query: str,
    vector_fields: str,
    semantic_configuration: str,
    vector_text: str,
    k: int = 50,
    top: int = 10,
    exhaustive: bool = False,
    weight: float = 0.0,
    select: str = "",
    filter: str = "",
    search_fields: str = "",
    query_caption: str = "extractive",
    query_answer: str = "",
    query_answer_count: int = 0,
    query_answer_threshold: float = 0.0,
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Run hybrid retrieval with semantic reranking.

    Parameters
    ----------
    index_name: str
        Target index name.
    query: str
        Natural-language or keyword query for lexical search.
    vector_fields: str
        Comma-separated vector field names.
    semantic_configuration: str
        Semantic configuration name defined on the index.
    vector_text: str
        Raw query text used for vectorization.
    k / top: int, optional
        Vector candidate count and final result count (top defaults to 10).
    exhaustive / weight: optional
        Controls for vector recall and weighting.
    query_caption / query_answer: Optional[str]
        Semantic caption and answer modes (`"extractive"`, `"summary"`, etc.).
    query_answer_count / query_answer_threshold: Optional
        Controls for the number of answers and confidence threshold.
    select / filter / search_fields: Optional[str]
        Additional shaping of results and filters.
    api_key / endpoint: Optional[str]
        Override default connection information.

    Returns
    -------
    dict
        Response containing `documents`, `count`, optional `answers`, `captions`, and continuation metadata.
    """
    # Convert empty strings and sentinel values to None
    weight = None if weight == 0.0 else weight
    select = None if select == "" else select
    filter = None if filter == "" else filter
    search_fields = None if search_fields == "" else search_fields
    # query_caption has default "extractive", don't convert
    query_answer = None if query_answer == "" else query_answer
    query_answer_count = None if query_answer_count == 0 else query_answer_count
    query_answer_threshold = None if query_answer_threshold == 0.0 else query_answer_threshold
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_key(api_key)

    vector_queries = _build_vector_query(
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=exhaustive,
        weight=weight,
    )

    search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "query_type": "semantic",
        "semantic_configuration_name": semantic_configuration,
        "top": top,
        "include_total_count": True,
    }
    if query_caption:
        search_kwargs["query_caption"] = query_caption
    if query_answer:
        search_kwargs["query_answer"] = query_answer
        if query_answer_count is not None:
            search_kwargs["query_answer_count"] = query_answer_count
        if query_answer_threshold is not None:
            search_kwargs["query_answer_threshold"] = query_answer_threshold
    if filter:
        search_kwargs["filter"] = filter
    if select:
        search_kwargs["select"] = _comma_split(select)
    if search_fields:
        search_kwargs["search_fields"] = _comma_split(search_fields)

    return await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=query,
        search_kwargs=search_kwargs,
    )


def _parse_key_value_configs(config_str: str) -> List[Dict[str, Any]]:
    if not config_str or not config_str.strip():
        return []

    # Split by semicolon for multiple sources
    source_entries = [entry.strip() for entry in config_str.split(";") if entry.strip()]

    sources = []
    for entry in source_entries:
        # Parse key-value pairs for a single source
        pairs = [pair.strip() for pair in entry.split(",") if pair.strip()]

        source_config: Dict[str, Any] = {}

        for pair in pairs:
            if "=" not in pair:
                raise ValueError(f"Invalid key-value pair: '{pair}'. Expected format: 'key=value'")

            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key or not value:
                raise ValueError(f"Empty key or value in pair: '{pair}'")

            # Type conversion logic
            # Boolean values
            if value.lower() in ("true", "false"):
                source_config[key] = value.lower() == "true"
            else:
                # Try numeric conversion
                try:
                    if "." in value:
                        source_config[key] = float(value)
                    else:
                        source_config[key] = int(value)
                except ValueError:
                    # Keep as string
                    source_config[key] = value

        # Validate required fields
        if "knowledgeSourceName" not in source_config:
            raise ValueError(f"Missing required 'knowledgeSourceName' in source config: '{entry}'")
        if "kind" not in source_config:
            raise ValueError(f"Missing required 'kind' in source config: '{entry}'")

        sources.append(source_config)

    return sources

@mcp.tool(
    name="multimodal_hybrid_search",
    description="Two-stage multimodal hybrid search: first retrieves text chunks with hybrid search + semantic reranker, then retrieves images from the same pages.",
)
async def multimodal_hybrid_search(
    index_name: str,
    query: str,
    vector_fields: str,
    semantic_configuration: str,
    vector_text: str,
    k: int = 50,
    top: int = 10,
    image_top: int = 10,
    exhaustive: bool = False,
    weight: float = 0.0,
    select: str = "",
    additional_filter: str = "",
    search_fields: str = "",
    query_caption: str = "extractive",
    query_answer: str = "",
    query_answer_count: int = 0,
    query_answer_threshold: float = 0.0,
    sharepoint_prefix: str = "",
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """Execute two-stage multimodal hybrid retrieval with semantic reranking.

    Stage 1: Text hybrid search (BM25 + Vector + Semantic Reranker)
    - Filter: chunk_type=text
    - Returns: All text results and extracts page numbers

    Stage 2: Image hybrid search (BM25 + Vector + Semantic Reranker)
    - Filter: chunk_type=image AND page_from IN (pages from stage 1)
    - Returns: All image results from the same pages

    Parameters
    ----------
    index_name: str
        Target index name.
    query: str
        Natural-language or keyword query for lexical search.
    vector_fields: str
        Comma-separated vector field names.
    semantic_configuration: str
        Semantic configuration name defined on the index.
    vector_text: str
        Raw query text used for vectorization.
    k / top: int, optional
        Vector candidate count and final text result count (top defaults to 10).
    image_top: int, optional
        Maximum number of image results to return from stage 2 (default 10).
    exhaustive / weight: optional
        Controls for vector recall and weighting.
    query_caption / query_answer: Optional[str]
        Semantic caption and answer modes (`"extractive"`, `"summary"`, etc.).
    query_answer_count / query_answer_threshold: Optional
        Controls for the number of answers and confidence threshold.
    select: Optional[str]
        Fields to return in results.
    additional_filter: Optional[str]
        Additional OData filter expression to apply to both stages.
    search_fields: Optional[str]
        Fields to search in.
    sharepoint_prefix: Optional[str]
        SharePoint URL prefix to replace the blob storage path prefix in source_path.
        If provided, the blob storage prefix will be replaced with this SharePoint prefix.
    api_key / endpoint: Optional[str]
        Override default connection information.

    Returns
    -------
    dict
        Response containing:
        - results: List of organized results, one per text document, each with:
            - page_from: Starting page number
            - page_to: Ending page number
            - source_document_path: Markdown-formatted document link ![filename](path)
            - text_content: Full text content
            - reranker_score: Semantic reranker score
            - search_score: Hybrid search score
            - images: Markdown-formatted image links (newline-separated)
        - total_pages: Number of result pages
        - text_count: Total text document count
        - image_count: Total image document count
        - answers: Optional semantic answers
        - captions: Optional semantic captions
    """
    # Convert empty strings and sentinel values to None for optional parameters
    weight = None if weight == 0.0 else weight
    select = None if select == "" else select
    additional_filter = None if additional_filter == "" else additional_filter
    search_fields = None if search_fields == "" else search_fields
    # query_caption has default "extractive", don't convert
    query_answer = None if query_answer == "" else query_answer
    query_answer_count = None if query_answer_count == 0 else query_answer_count
    query_answer_threshold = None if query_answer_threshold == 0.0 else query_answer_threshold
    sharepoint_prefix = None if sharepoint_prefix == "" else sharepoint_prefix
    api_key_param = None if api_key == "" else api_key
    endpoint_param = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint_param)
    key = _resolve_key(api_key_param)

    vector_queries = _build_vector_query(
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=exhaustive,
        weight=weight,
    )

    # Stage 1: Text hybrid search with semantic reranking
    text_filter = "chunk_type eq 'text'"
    if additional_filter:
        text_filter = f"({text_filter}) and ({additional_filter})"

    text_search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "query_type": "semantic",
        "semantic_configuration_name": semantic_configuration,
        "top": top,
        "include_total_count": True,
        "filter": text_filter,
    }
    if query_caption:
        text_search_kwargs["query_caption"] = query_caption
    if query_answer:
        text_search_kwargs["query_answer"] = query_answer
        if query_answer_count is not None:
            text_search_kwargs["query_answer_count"] = query_answer_count
        if query_answer_threshold is not None:
            text_search_kwargs["query_answer_threshold"] = query_answer_threshold
    if select:
        text_search_kwargs["select"] = _comma_split(select)
    if search_fields:
        text_search_kwargs["search_fields"] = _comma_split(search_fields)

    logger.info("Stage 1: Executing text hybrid search with filter: %s", text_filter)
    text_result = await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=query,
        search_kwargs=text_search_kwargs,
    )

    # Extract page numbers from text results
    text_documents = text_result.get("documents", [])
    page_numbers = _extract_page_numbers_from_documents(text_documents)
    logger.info("Stage 1 completed: Found %d text documents across %d pages: %s",
                len(text_documents), len(page_numbers), page_numbers)

    # Stage 2: Image hybrid search filtered by pages from stage 1
    image_result = {"documents": [], "count": 0}
    if page_numbers:
        # Build filter for images on the same pages
        page_filter_parts = [f"page_from eq {page}" for page in page_numbers]
        image_filter = f"(chunk_type eq 'image') and ({' or '.join(page_filter_parts)})"
        if additional_filter:
            image_filter = f"({image_filter}) and ({additional_filter})"

        image_search_kwargs: Dict[str, Any] = {
            "vector_queries": vector_queries,
            "query_type": "semantic",
            "semantic_configuration_name": semantic_configuration,
            "top": image_top,
            "include_total_count": True,
            "filter": image_filter,
        }
        if select:
            image_search_kwargs["select"] = _comma_split(select)
        if search_fields:
            image_search_kwargs["search_fields"] = _comma_split(search_fields)

        logger.info("Stage 2: Executing image hybrid search with filter: %s", image_filter)
        image_result = await _execute_search(
            endpoint=resolved_endpoint,
            key=key,
            index_name=index_name,
            search_text=query,
            search_kwargs=image_search_kwargs,
        )
        logger.info("Stage 2 completed: Found %d image documents", len(image_result.get("documents", [])))
    else:
        logger.info("Stage 2 skipped: No pages found in text results")

    # Organize results by page - one result per text document
    def _organize_multimodal_results(
        text_docs: List[Dict[str, Any]],
        image_docs: List[Dict[str, Any]],
        sharepoint_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Organize text and image results by page, one result per text document."""

        def _replace_path_prefix(original_path: str, new_prefix: Optional[str]) -> str:
            """Replace blob storage path prefix with SharePoint prefix and ensure proper URL encoding."""
            if not new_prefix or not original_path:
                return original_path

            # Extract filename from the original path (already URL-encoded)
            filename = original_path.split("/")[-1] if "/" in original_path else original_path

            # Parse and encode the new prefix
            # Split into protocol, domain, and path parts
            if "://" in new_prefix:
                protocol_domain, path_part = new_prefix.split("://", 1)
                protocol = protocol_domain + "://"

                # Split domain and path
                if "/" in path_part:
                    domain, path = path_part.split("/", 1)
                    # Decode first to handle both encoded and unencoded input, then encode
                    path_segments = path.rstrip("/").split("/")
                    encoded_segments = [quote(unquote(seg), safe="") for seg in path_segments]
                    encoded_path = "/".join(encoded_segments)
                    new_prefix = f"{protocol}{domain}/{encoded_path}/"
                else:
                    new_prefix = f"{protocol}{path_part}/"
            else:
                # If no protocol, just ensure it ends with '/'
                if not new_prefix.endswith("/"):
                    new_prefix += "/"

            return new_prefix + filename

        # Group images by page
        images_by_page = {}
        for doc in image_docs:
            page_from = doc.get("page_from") or doc.get("pageNumberFrom")
            if page_from is None:
                continue

            if page_from not in images_by_page:
                images_by_page[page_from] = []

            # Try multiple possible image path fields
            source_path = doc.get("imagePath") or doc.get("source_path") or doc.get("image_path")
            if source_path:
                # Images keep original blob path (no prefix replacement)
                # Extract filename for alt text
                filename = source_path.split("/")[-1] if "/" in source_path else source_path
                images_by_page[page_from].append(f"![{filename}]({source_path})")

        # Create one result per text document
        results = []
        for doc in text_docs:
            page_from = doc.get("page_from") or doc.get("pageNumberFrom")
            page_to = doc.get("page_to") or doc.get("pageNumberTo")

            if page_from is None:
                continue

            # Format source document path as Markdown link
            source_document = doc.get("source_document", "")
            source_path = doc.get("source_path", "")

            # Replace prefix if sharepoint_prefix is provided
            processed_doc_path = _replace_path_prefix(source_path, sharepoint_prefix)
            # Add page anchor to the URL (SharePoint format: #page=N)
            if processed_doc_path:
                processed_doc_path_with_page = f"{processed_doc_path}#page={page_from}"
                # Document link uses [text](url), not ![text](url) which is for images
                source_document_path = f"[{source_document}]({processed_doc_path_with_page})"
            else:
                source_document_path = source_document

            result = {
                "page_from": page_from,
                "page_to": page_to,
                "source_document_path": source_document_path,
                "text_content": doc.get("content"),
                "reranker_score": doc.get("@search.reranker_score"),
                "search_score": doc.get("@search.score"),
                "images": "\n".join(images_by_page.get(page_from, []))
            }
            results.append(result)

        # Sort by page number
        return sorted(results, key=lambda x: x["page_from"])

    organized_results = _organize_multimodal_results(
        text_documents,
        image_result.get("documents", []),
        sharepoint_prefix
    )

    # Combine results
    combined_result: Dict[str, Any] = {
        "results": organized_results,
        "total_pages": len(organized_results),
        "text_count": text_result.get("count", 0),
        "image_count": image_result.get("count", 0),
    }

    # Include answers and captions from text search if present
    if "answers" in text_result:
        combined_result["answers"] = text_result["answers"]
    if "captions" in text_result:
        combined_result["captions"] = text_result["captions"]

    return combined_result


@mcp.tool(
    name="agentic_retrieval",
    description="Run Azure AI Search agentic retrieval pipeline with flexible knowledge source configuration (2025-11-01-preview API).",
)
async def agentic_retrieval(
    knowledge_base_name: str,
    query: str,
    intent_query: str = "",
    reasoning_effort: str = "",
    output_mode: str = "answerSynthesis",
    include_activity: bool = True,
    max_runtime_seconds: int = 0,
    max_output_size: int = 0,
    knowledge_source_configs: str = "",
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """调用 Azure AI Search Knowledge Base 的 Agentic Retrieval 接口。

    参数:
        knowledge_base_name: 要查询的 knowledge base 名称。
        query: 用户问题或查询文本，不能为空。
        intent_query: 可选的显式检索意图，用于绕过默认的查询规划。
        reasoning_effort: 检索推理强度，可选值为 minimal、low、medium。
        output_mode: 输出模式，支持 answerSynthesis 或 extractedData。
        include_activity: 是否在结果中返回 activity 调试信息。
        max_runtime_seconds: 服务端允许的最大运行时长（秒）。
        max_output_size: 输出 token 上限。
        knowledge_source_configs: 知识源配置字符串，使用 key=value 形式并以分号分隔多个来源。
        api_key: 可选的管理员密钥覆盖值。
        endpoint: 可选的 Azure Search 终结点覆盖值。

    返回:
        Dict[str, Any]: 格式化后的检索结果，包含 answer、references 以及 _status_code。
    """
    # Convert empty strings and sentinel values to None
    intent_query = None if intent_query == "" else intent_query
    reasoning_effort = None if reasoning_effort == "" else reasoning_effort
    max_runtime_seconds = None if max_runtime_seconds == 0 else max_runtime_seconds
    max_output_size = None if max_output_size == 0 else max_output_size
    knowledge_source_configs = None if knowledge_source_configs == "" else knowledge_source_configs
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_admin_key(api_key)

    request_body: Dict[str, Any] = {
        "includeActivity": include_activity,
        "outputMode": output_mode,
    }

    if not query:
        raise ValueError("`query` must be provided for agentic retrieval requests.")
    request_body["messages"] = _build_messages_from_query(query)

    if intent_query:
        request_body.setdefault("intents", []).append({"type": "semantic", "search": intent_query})

    if reasoning_effort:
        effort_kind = reasoning_effort.lower()
        if effort_kind not in {"minimal", "low", "medium"}:
            raise ValueError("reasoning_effort must be one of: minimal, low, medium")
        request_body["retrievalReasoningEffort"] = {"kind": effort_kind}

    if max_runtime_seconds is not None:
        request_body["maxRuntimeInSeconds"] = max_runtime_seconds
    if max_output_size is not None:
        request_body["maxOutputSize"] = max_output_size

    # Parse knowledge source configurations using key-value format
    if knowledge_source_configs:
        try:
            sources = _parse_key_value_configs(knowledge_source_configs)
            request_body["knowledgeSourceParams"] = sources
        except ValueError as exc:
            raise ValueError(f"Failed to parse knowledge_source_configs: {exc}") from exc

    url = (
        f"{resolved_endpoint}/knowledgebases('{quote(knowledge_base_name, safe='')}')/retrieve"
        f"?api-version={AGENTIC_API_VERSION}"
    )

    timeout_budget: int = _build_agentic_timeout_budget(max_runtime_seconds)
    headers = {
        "api-key": key,
        "Content-Type": "application/json",
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_budget)) as session:
            async with session.post(url, headers=headers, json=request_body) as resp:
                text = await resp.text()
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Agentic retrieval failed ({resp.status}): {text}")
                try:
                    raw_data = json.loads(text)
                except json.JSONDecodeError:
                    raw_data = {"raw": text}

                # Format the response
                formatted_data = _format_agentic_response(raw_data)
                formatted_data["_status_code"] = resp.status
                return formatted_data
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Agentic retrieval timed out while waiting for Azure AI Search to respond. "
            f"Client timeout budget={timeout_budget}s. "
            "Increase AZURE_SEARCH_AGENTIC_TIMEOUT, increase AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER, "
            "or reduce request complexity/max_runtime_seconds for long-running answerSynthesis or multi-source queries."
        ) from exc


def main() -> None:
    global AZURE_SEARCH_ENDPOINT
    parser = ArgumentParser(description="Start the Azure AI Search MCP server")
    parser.add_argument(
        "--transport",
        required=False,
        default="stdio",
        choices=("stdio", "sse", "streamable-http"),
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--endpoint",
        required=False,
        default=AZURE_SEARCH_ENDPOINT,
        help="Override the Azure AI Search endpoint",
    )
    args = parser.parse_args()

    logger.info("Starting Azure AI Search MCP server with transport=%s", args.transport)
    resolved_endpoint = _resolve_endpoint(args.endpoint)
    if resolved_endpoint != AZURE_SEARCH_ENDPOINT:
        os.environ["AZURE_SEARCH_ENDPOINT"] = resolved_endpoint
        AZURE_SEARCH_ENDPOINT = resolved_endpoint
    logger.info("Azure Search endpoint resolved to %s", resolved_endpoint)

    mcp.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
