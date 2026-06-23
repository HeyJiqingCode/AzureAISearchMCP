import asyncio
import io
import logging
import os
import re
import sys
from argparse import ArgumentParser
from typing import Any, Dict, Final, List, Optional

from azure.core.exceptions import HttpResponseError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.knowledgebases.aio import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    KnowledgeRetrievalLowReasoningEffort,
    KnowledgeRetrievalMediumReasoningEffort,
    KnowledgeRetrievalMinimalReasoningEffort,
    KnowledgeRetrievalSemanticIntent,
)
from azure.search.documents.models import (
    HybridSearch,
    SearchScoreThreshold,
    VectorizableTextQuery,
    VectorSimilarityThreshold,
)
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
DEFAULT_MCP_HOST: Final[str] = os.getenv("MCP_HOST", "0.0.0.0")
DEFAULT_MCP_PORT: Final[int] = int(os.getenv("MCP_PORT", "8000"))


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


# 从 Azure SDK 异常中提取 HTTP 状态码，避免不同异常形态输出 None。
def _http_status_code(exc: HttpResponseError) -> Any:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code
    response = getattr(exc, "response", None)
    if response is None:
        return "unknown"
    return getattr(response, "status_code", None) or getattr(response, "status", "unknown")


# 将 SDK 模型和嵌套容器转换为 MCP 可返回的基础类型。
def _to_plain_data(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return _to_plain_data(value.as_dict())
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


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
        int: 供 SDK 调用使用的总超时秒数，包含服务端运行时间和额外缓冲。
    """
    requested_runtime_seconds: int = max_runtime_seconds or 0
    buffered_runtime_seconds: int = requested_runtime_seconds + AGENTIC_TIMEOUT_BUFFER_SECONDS
    return max(AGENTIC_HTTP_TIMEOUT_SECONDS, buffered_runtime_seconds)


def _normalize_document(document: Dict[str, Any]) -> Dict[str, Any]:
    return _to_plain_data(document)


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


# 汇总 Azure Search 分页结果，并尽量保留 count、answer、facet 等附加元数据。
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
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read Azure Search result metadata via %s: %s", attr, exc)
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
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read Azure Search continuation token: %s", exc)
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


# 执行统一的 SearchClient 查询，并为客户排障补充索引和查询上下文。
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
    try:
        async with client:
            result_pager = await client.search(
                search_text=search_text,
                **search_kwargs,
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            payload = await _collect_results(result_pager)
    except HttpResponseError as exc:
        query_label = search_text if search_text is not None else "<vector-only>"
        status_code = _http_status_code(exc)
        raise RuntimeError(
            "Azure AI Search query failed "
            f"(index={index_name}, query={query_label}, status={status_code}): {exc.message}"
        ) from exc
    return payload


def _build_vector_query(
    *,
    vector_text: str,
    vector_fields: Optional[str],
    k: int,
    exhaustive: bool,
    weight: Optional[float] = None,
    oversampling: Optional[float] = None,
    filter_override: Optional[str] = None,
    vector_similarity_threshold: Optional[float] = None,
    search_score_threshold: Optional[float] = None,
) -> List[VectorizableTextQuery]:
    if vector_fields is None:
        raise ValueError("vector_fields must be provided for vector-enabled search.")
    if vector_similarity_threshold is not None and search_score_threshold is not None:
        raise ValueError("Only one of vector_similarity_threshold or search_score_threshold can be provided.")

    vector_query = VectorizableTextQuery(
        text=vector_text,
        fields=vector_fields,
        k_nearest_neighbors=k,
        exhaustive=exhaustive,
    )
    if weight is not None:
        vector_query.weight = weight
    if oversampling is not None:
        vector_query.oversampling = oversampling
    if filter_override:
        vector_query.filter_override = filter_override
    if vector_similarity_threshold is not None:
        vector_query.threshold = VectorSimilarityThreshold(value=vector_similarity_threshold)
    if search_score_threshold is not None:
        vector_query.threshold = SearchScoreThreshold(value=search_score_threshold)
    return [vector_query]


# 构造 hybridSearch 请求对象，未设置时不向服务端发送该配置。
def _build_hybrid_search(
    *,
    max_text_recall_size: Optional[int],
    count_and_facet_mode: Optional[str],
) -> Optional[HybridSearch]:
    if max_text_recall_size is None and not count_and_facet_mode:
        return None
    return HybridSearch(
        max_text_recall_size=max_text_recall_size,
        count_and_facet_mode=count_and_facet_mode,
    )


# 根据工具参数选择 Knowledge Base retrieval 的推理强度模型。
def _build_reasoning_effort(reasoning_effort: Optional[str]) -> Optional[Any]:
    if not reasoning_effort:
        return None

    effort_kind = reasoning_effort.lower()
    if effort_kind == "minimal":
        return KnowledgeRetrievalMinimalReasoningEffort()
    if effort_kind == "low":
        return KnowledgeRetrievalLowReasoningEffort()
    if effort_kind == "medium":
        return KnowledgeRetrievalMediumReasoningEffort()
    raise ValueError("reasoning_effort must be one of: minimal, low, medium")


def _format_agentic_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """整理 Knowledge Base Retrieval 响应，同时保留原始结构。"""
    try:
        answer_text = raw_response["response"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        answer_text = ""

    def replace_ref(match):
        ref_id = int(match.group(1))
        superscript_num = ref_id + 1
        return f"<sup>{superscript_num}</sup>"

    formatted_answer = re.sub(r'\[ref_id:(\d+)\]', replace_ref, answer_text)

    activity_map = {}
    activities = raw_response.get("activity", [])
    for activity in activities:
        activity_id = activity.get("id")
        knowledge_source_name = activity.get("knowledgeSourceName")
        if activity_id is not None and knowledge_source_name:
            activity_map[activity_id] = knowledge_source_name

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
            title = ref.get("title", "Untitled")
            url = ref.get("url", "")
            formatted_refs.append(f"{display_id}. [Web] [{title}]({url})")

        elif ref_type in ["searchIndex", "remoteSharePoint", "azureBlob", "indexedOneLake"]:
            title = ref.get("title", "Untitled")
            reranker_score = ref.get("rerankerScore")
            score_text = f"{reranker_score:.2f}" if isinstance(reranker_score, (int, float)) else "n/a"
            formatted_refs.append(
                f"{display_id}. [KnowledgeBase: {knowledge_source_name}] [rerankerScore: {score_text}] Title: {title}"
            )
        else:
            formatted_refs.append(f"{display_id}. [Unknown Type: {ref_type}]")

    def _reference_sort_key(value: str) -> int:
        try:
            return int(value.split(".", 1)[0])
        except (ValueError, IndexError):
            return 0

    formatted_refs.sort(key=_reference_sort_key)

    return {
        "answer": formatted_answer,
        "references": formatted_refs,
        "response": raw_response.get("response", []),
        "activity": raw_response.get("activity", []),
        "raw_references": raw_response.get("references", []),
        "metadata": raw_response.get("metadata", {}),
    }

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
    semantic_query: str = "",
    query_caption: str = "extractive",
    query_caption_highlight_enabled: bool = True,
    query_answer: str = "",
    query_answer_count: int = 0,
    query_answer_threshold: float = 0.0,
    semantic_error_mode: str = "",
    semantic_max_wait_in_milliseconds: int = 0,
    debug: str = "",
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
    semantic_query = None if semantic_query == "" else semantic_query
    # query_caption has default "extractive", don't convert to None
    query_answer = None if query_answer == "" else query_answer
    query_answer_count = None if query_answer_count == 0 else query_answer_count
    query_answer_threshold = None if query_answer_threshold == 0.0 else query_answer_threshold
    semantic_error_mode = None if semantic_error_mode == "" else semantic_error_mode
    semantic_max_wait_in_milliseconds = (
        None if semantic_max_wait_in_milliseconds == 0 else semantic_max_wait_in_milliseconds
    )
    debug = None if debug == "" else debug

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
    if semantic_query:
        search_kwargs["semantic_query"] = semantic_query
    if query_caption:
        search_kwargs["query_caption"] = query_caption
        search_kwargs["query_caption_highlight_enabled"] = query_caption_highlight_enabled
    if query_answer:
        search_kwargs["query_answer"] = query_answer
        if query_answer_count is not None:
            search_kwargs["query_answer_count"] = query_answer_count
        if query_answer_threshold is not None:
            search_kwargs["query_answer_threshold"] = query_answer_threshold
    if semantic_error_mode:
        search_kwargs["semantic_error_mode"] = semantic_error_mode
    if semantic_max_wait_in_milliseconds is not None:
        search_kwargs["semantic_max_wait_in_milliseconds"] = semantic_max_wait_in_milliseconds
    if debug:
        search_kwargs["debug"] = debug

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
    oversampling: float = 0.0,
    filter_override: str = "",
    vector_filter_mode: str = "",
    vector_similarity_threshold: float = 0.0,
    search_score_threshold: float = 0.0,
    select: str = "",
    filter: str = "",
    debug: str = "",
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
    oversampling = None if oversampling == 0.0 else oversampling
    filter_override = None if filter_override == "" else filter_override
    vector_filter_mode = None if vector_filter_mode == "" else vector_filter_mode
    vector_similarity_threshold = None if vector_similarity_threshold == 0.0 else vector_similarity_threshold
    search_score_threshold = None if search_score_threshold == 0.0 else search_score_threshold
    select = None if select == "" else select
    filter = None if filter == "" else filter
    debug = None if debug == "" else debug
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
        oversampling=oversampling,
        filter_override=filter_override,
        vector_similarity_threshold=vector_similarity_threshold,
        search_score_threshold=search_score_threshold,
    )

    search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "top": k,
        "include_total_count": True,
    }
    if vector_filter_mode:
        search_kwargs["vector_filter_mode"] = vector_filter_mode
    if select:
        search_kwargs["select"] = _comma_split(select)
    if filter:
        search_kwargs["filter"] = filter
    if debug:
        search_kwargs["debug"] = debug

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
    oversampling: float = 0.0,
    filter_override: str = "",
    vector_filter_mode: str = "",
    vector_similarity_threshold: float = 0.0,
    search_score_threshold: float = 0.0,
    max_text_recall_size: int = 0,
    count_and_facet_mode: str = "",
    select: str = "",
    filter: str = "",
    search_fields: str = "",
    debug: str = "",
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
    oversampling = None if oversampling == 0.0 else oversampling
    filter_override = None if filter_override == "" else filter_override
    vector_filter_mode = None if vector_filter_mode == "" else vector_filter_mode
    vector_similarity_threshold = None if vector_similarity_threshold == 0.0 else vector_similarity_threshold
    search_score_threshold = None if search_score_threshold == 0.0 else search_score_threshold
    max_text_recall_size = None if max_text_recall_size == 0 else max_text_recall_size
    count_and_facet_mode = None if count_and_facet_mode == "" else count_and_facet_mode
    select = None if select == "" else select
    filter = None if filter == "" else filter
    search_fields = None if search_fields == "" else search_fields
    debug = None if debug == "" else debug
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
        oversampling=oversampling,
        filter_override=filter_override,
        vector_similarity_threshold=vector_similarity_threshold,
        search_score_threshold=search_score_threshold,
    )
    hybrid_search_config = _build_hybrid_search(
        max_text_recall_size=max_text_recall_size,
        count_and_facet_mode=count_and_facet_mode,
    )

    search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "top": top,
        "include_total_count": True,
    }
    if hybrid_search_config:
        search_kwargs["hybrid_search"] = hybrid_search_config
    if vector_filter_mode:
        search_kwargs["vector_filter_mode"] = vector_filter_mode
    if select:
        search_kwargs["select"] = _comma_split(select)
    if filter:
        search_kwargs["filter"] = filter
    if search_fields:
        search_kwargs["search_fields"] = _comma_split(search_fields)
    if debug:
        search_kwargs["debug"] = debug

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
    oversampling: float = 0.0,
    filter_override: str = "",
    vector_filter_mode: str = "",
    vector_similarity_threshold: float = 0.0,
    search_score_threshold: float = 0.0,
    max_text_recall_size: int = 0,
    count_and_facet_mode: str = "",
    select: str = "",
    filter: str = "",
    search_fields: str = "",
    semantic_query: str = "",
    query_caption: str = "extractive",
    query_caption_highlight_enabled: bool = True,
    query_answer: str = "",
    query_answer_count: int = 0,
    query_answer_threshold: float = 0.0,
    semantic_error_mode: str = "",
    semantic_max_wait_in_milliseconds: int = 0,
    debug: str = "",
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
    oversampling = None if oversampling == 0.0 else oversampling
    filter_override = None if filter_override == "" else filter_override
    vector_filter_mode = None if vector_filter_mode == "" else vector_filter_mode
    vector_similarity_threshold = None if vector_similarity_threshold == 0.0 else vector_similarity_threshold
    search_score_threshold = None if search_score_threshold == 0.0 else search_score_threshold
    max_text_recall_size = None if max_text_recall_size == 0 else max_text_recall_size
    count_and_facet_mode = None if count_and_facet_mode == "" else count_and_facet_mode
    select = None if select == "" else select
    filter = None if filter == "" else filter
    search_fields = None if search_fields == "" else search_fields
    semantic_query = None if semantic_query == "" else semantic_query
    # query_caption has default "extractive", don't convert
    query_answer = None if query_answer == "" else query_answer
    query_answer_count = None if query_answer_count == 0 else query_answer_count
    query_answer_threshold = None if query_answer_threshold == 0.0 else query_answer_threshold
    semantic_error_mode = None if semantic_error_mode == "" else semantic_error_mode
    semantic_max_wait_in_milliseconds = (
        None if semantic_max_wait_in_milliseconds == 0 else semantic_max_wait_in_milliseconds
    )
    debug = None if debug == "" else debug
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
        oversampling=oversampling,
        filter_override=filter_override,
        vector_similarity_threshold=vector_similarity_threshold,
        search_score_threshold=search_score_threshold,
    )
    hybrid_search_config = _build_hybrid_search(
        max_text_recall_size=max_text_recall_size,
        count_and_facet_mode=count_and_facet_mode,
    )

    search_kwargs: Dict[str, Any] = {
        "vector_queries": vector_queries,
        "query_type": "semantic",
        "semantic_configuration_name": semantic_configuration,
        "top": top,
        "include_total_count": True,
    }
    if hybrid_search_config:
        search_kwargs["hybrid_search"] = hybrid_search_config
    if vector_filter_mode:
        search_kwargs["vector_filter_mode"] = vector_filter_mode
    if semantic_query:
        search_kwargs["semantic_query"] = semantic_query
    if query_caption:
        search_kwargs["query_caption"] = query_caption
        search_kwargs["query_caption_highlight_enabled"] = query_caption_highlight_enabled
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
    if semantic_error_mode:
        search_kwargs["semantic_error_mode"] = semantic_error_mode
    if semantic_max_wait_in_milliseconds is not None:
        search_kwargs["semantic_max_wait_in_milliseconds"] = semantic_max_wait_in_milliseconds
    if debug:
        search_kwargs["debug"] = debug

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
    name="agentic_retrieval",
    description="Run Azure AI Search Knowledge Base retrieval through the latest Python SDK preview shape.",
)
async def agentic_retrieval(
    knowledge_base_name: str,
    query: str,
    intent_query: str = "",
    reasoning_effort: str = "low",
    output_mode: str = "answerSynthesis",
    include_activity: bool = True,
    max_runtime_seconds: int = 0,
    max_output_size: int = 0,
    max_output_documents: int = 0,
    knowledge_source_configs: str = "",
    query_source_authorization: str = "",
    api_key: str = "",
    endpoint: str = "",
) -> Dict[str, Any]:
    """调用 Azure AI Search Knowledge Base Retrieval SDK。"""
    # Convert empty strings and sentinel values to None
    intent_query = None if intent_query == "" else intent_query
    reasoning_effort = None if reasoning_effort == "" else reasoning_effort
    output_mode = None if output_mode == "" else output_mode
    max_runtime_seconds = None if max_runtime_seconds == 0 else max_runtime_seconds
    max_output_size = None if max_output_size == 0 else max_output_size
    max_output_documents = None if max_output_documents == 0 else max_output_documents
    knowledge_source_configs = None if knowledge_source_configs == "" else knowledge_source_configs
    query_source_authorization = None if query_source_authorization == "" else query_source_authorization
    api_key = None if api_key == "" else api_key
    endpoint = None if endpoint == "" else endpoint

    resolved_endpoint = _resolve_endpoint(endpoint)
    key = _resolve_admin_key(api_key)

    if not query:
        raise ValueError("`query` must be provided for agentic retrieval requests.")

    reasoning = _build_reasoning_effort(reasoning_effort)
    use_intent = (reasoning_effort or "").lower() == "minimal" or intent_query is not None
    if use_intent and output_mode == "answerSynthesis":
        raise ValueError("answerSynthesis requires message-based retrieval. Use output_mode=extractedData for minimal intent retrieval.")

    request_kwargs: Dict[str, Any] = {
        "include_activity": include_activity,
    }
    if use_intent:
        request_kwargs["intents"] = [
            KnowledgeRetrievalSemanticIntent(search=intent_query or query)
        ]
    else:
        request_kwargs["messages"] = [
            KnowledgeBaseMessage(
                role=message["role"],
                content=[
                    KnowledgeBaseMessageTextContent(text=part["text"])
                    for part in message["content"]
                    if part.get("type") == "text"
                ],
            )
            for message in _build_messages_from_query(query)
        ]
    if reasoning:
        request_kwargs["retrieval_reasoning_effort"] = reasoning
    if output_mode:
        request_kwargs["output_mode"] = output_mode
    if max_runtime_seconds is not None:
        request_kwargs["max_runtime_in_seconds"] = max_runtime_seconds
    if max_output_size is not None:
        request_kwargs["max_output_size"] = max_output_size
    if max_output_documents is not None:
        request_kwargs["max_output_documents"] = max_output_documents
    if knowledge_source_configs:
        try:
            request_kwargs["knowledge_source_params"] = _parse_key_value_configs(knowledge_source_configs)
        except ValueError as exc:
            raise ValueError(f"Failed to parse knowledge_source_configs: {exc}") from exc

    request = KnowledgeBaseRetrievalRequest(**request_kwargs)

    timeout_budget: int = _build_agentic_timeout_budget(max_runtime_seconds)
    client = KnowledgeBaseRetrievalClient(
        endpoint=resolved_endpoint,
        credential=AzureKeyCredential(key),
        knowledge_base_name=knowledge_base_name,
    )

    try:
        async with client:
            retrieve_kwargs: Dict[str, Any] = {"retrieval_request": request}
            if query_source_authorization:
                retrieve_kwargs["query_source_authorization"] = query_source_authorization
            result = await asyncio.wait_for(
                client.retrieve(**retrieve_kwargs),
                timeout=timeout_budget,
            )
        raw_data = _to_plain_data(result)
        formatted_data = _format_agentic_response(raw_data)
        formatted_data["request"] = request.as_dict()
        return formatted_data
    except HttpResponseError as exc:
        raise RuntimeError(f"Agentic retrieval failed ({_http_status_code(exc)}): {exc.message}") from exc
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "Agentic retrieval timed out while waiting for Azure AI Search to respond. "
            f"Client timeout budget={timeout_budget}s. "
            "Increase AZURE_SEARCH_AGENTIC_TIMEOUT, increase AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER, "
            "or reduce request complexity/max_runtime_seconds for long-running answerSynthesis or multi-source queries."
        ) from exc


# 解析 MCP 启动参数，让本地 stdio 和远程 HTTP 部署使用同一个入口。
def main() -> None:
    global AZURE_SEARCH_ENDPOINT
    parser = ArgumentParser(description="Start the Azure AI Search MCP server")
    parser.add_argument(
        "--transport",
        required=False,
        default="stdio",
        choices=("stdio", "http", "streamable-http", "sse"),
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host",
        required=False,
        default=DEFAULT_MCP_HOST,
        help="Host for HTTP/SSE transports (default: MCP_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        required=False,
        default=DEFAULT_MCP_PORT,
        type=int,
        help="Port for HTTP/SSE transports (default: MCP_PORT or 8000)",
    )
    parser.add_argument(
        "--endpoint",
        required=False,
        default=None,
        help="Set a default Azure AI Search endpoint for this server process",
    )
    args = parser.parse_args()

    if args.endpoint:
        resolved_endpoint = _resolve_endpoint(args.endpoint)
        os.environ["AZURE_SEARCH_ENDPOINT"] = resolved_endpoint
        AZURE_SEARCH_ENDPOINT = resolved_endpoint
        logger.info("Azure Search endpoint resolved to %s", resolved_endpoint)
    elif AZURE_SEARCH_ENDPOINT:
        logger.info("Azure Search endpoint resolved to %s", _resolve_endpoint(AZURE_SEARCH_ENDPOINT))
    else:
        logger.warning(
            "Azure Search endpoint is not configured. The server can start, "
            "but each tool call must pass endpoint or set AZURE_SEARCH_ENDPOINT."
        )

    transport_kwargs: Dict[str, Any] = {}
    if args.transport != "stdio":
        transport_kwargs = {"host": args.host, "port": args.port}

    logger.info(
        "Starting Azure AI Search MCP server with transport=%s%s",
        args.transport,
        f" on {args.host}:{args.port}" if transport_kwargs else "",
    )
    mcp.run(transport=args.transport, **transport_kwargs)


if __name__ == "__main__":
    main()
