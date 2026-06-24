import asyncio
import io
import json
import logging
import os
import re
import sys
import time
from argparse import ArgumentParser
from typing import Any, Dict, Final, List, Optional, Tuple

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


def _none_if_empty(value: Optional[str]) -> Optional[str]:
    if value == "":
        return None
    return value


def _none_if_zero(value: Any) -> Any:
    if value == 0 or value == 0.0:
        return None
    return value


# 只在参数有效时写入 SDK kwargs，避免向服务端发送空字符串哨兵值。
def _add_if_present(target: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None and value != "":
        target[key] = value


def _add_csv_if_present(target: Dict[str, Any], key: str, value: Optional[str]) -> None:
    parts = _comma_split(value)
    if parts:
        target[key] = parts


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
COMMON_ADVANCED_OPTION_KEYS: Final[frozenset[str]] = frozenset({"debug"})
SEMANTIC_ADVANCED_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "semantic_query",
        "query_answer_count",
        "query_answer_threshold",
        "semantic_error_mode",
        "semantic_max_wait_in_milliseconds",
    }
)
VECTOR_ADVANCED_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "exhaustive",
        "weight",
        "oversampling",
        "filter_override",
        "vector_filter_mode",
        "vector_similarity_threshold",
        "search_score_threshold",
    }
)
HYBRID_ADVANCED_OPTION_KEYS: Final[frozenset[str]] = frozenset(
    {"max_text_recall_size", "count_and_facet_mode"}
)


def _resolve_endpoint(endpoint: Optional[str] = None) -> str:
    resolved = endpoint or AZURE_SEARCH_ENDPOINT
    if not resolved:
        raise RuntimeError(
            "Azure Search endpoint is not configured. Set AZURE_SEARCH_ENDPOINT or start the server with --endpoint."
        )
    return resolved.rstrip("/")


def _resolve_key(explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    if DEFAULT_QUERY_KEY:
        return DEFAULT_QUERY_KEY
    raise RuntimeError("Azure Search query key is not configured. Set AZURE_SEARCH_QUERY_KEY.")


def _resolve_admin_key(explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key
    if DEFAULT_ADMIN_KEY:
        return DEFAULT_ADMIN_KEY
    raise RuntimeError(
        "Agentic retrieval requires an admin key. Set AZURE_SEARCH_ADMIN_KEY."
    )


# 统一解析单次工具调用的连接信息，普通查询用 query key，Agentic 用 admin key。
def _resolve_call_context(*, admin: bool = False) -> Tuple[str, str]:
    resolved_endpoint = _resolve_endpoint()
    key = _resolve_admin_key(None) if admin else _resolve_key(None)
    return resolved_endpoint, key


# 解析 JSON 字符串形式的高级参数，并用白名单约束客户可配置范围。
def _parse_json_options(options: str, *, allowed_keys: frozenset[str], label: str) -> Dict[str, Any]:
    if not options.strip():
        return {}

    try:
        parsed = json.loads(options)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object.") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")

    unknown_keys = sorted(set(parsed) - set(allowed_keys))
    if unknown_keys:
        allowed = ", ".join(sorted(allowed_keys))
        unknown = ", ".join(unknown_keys)
        raise ValueError(f"Unsupported {label} keys: {unknown}. Allowed keys: {allowed}.")

    return parsed


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
        items.append(_to_plain_data(dict(item)))

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
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
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


# 添加 SearchClient.search 的通用可选参数，保持各工具只声明自身差异。
def _add_common_search_options(
    search_kwargs: Dict[str, Any],
    *,
    select: Optional[str] = None,
    filter: Optional[str] = None,
    search_fields: Optional[str] = None,
    debug: Optional[str] = None,
) -> None:
    _add_csv_if_present(search_kwargs, "select", select)
    _add_if_present(search_kwargs, "filter", filter)
    _add_csv_if_present(search_kwargs, "search_fields", search_fields)
    _add_if_present(search_kwargs, "debug", debug)


# 添加语义检索参数，供 semantic 和 semantic_hybrid 两类工具共享。
def _add_semantic_options(
    search_kwargs: Dict[str, Any],
    *,
    semantic_query: Optional[str],
    query_caption: Optional[str],
    query_caption_highlight_enabled: bool,
    query_answer: Optional[str],
    query_answer_count: Optional[int],
    query_answer_threshold: Optional[float],
    semantic_error_mode: Optional[str],
    semantic_max_wait_in_milliseconds: Optional[int],
) -> None:
    _add_if_present(search_kwargs, "semantic_query", semantic_query)
    if query_caption:
        search_kwargs["query_caption"] = query_caption
        search_kwargs["query_caption_highlight_enabled"] = query_caption_highlight_enabled
    if query_answer:
        search_kwargs["query_answer"] = query_answer
        _add_if_present(search_kwargs, "query_answer_count", query_answer_count)
        _add_if_present(search_kwargs, "query_answer_threshold", query_answer_threshold)
    _add_if_present(search_kwargs, "semantic_error_mode", semantic_error_mode)
    _add_if_present(search_kwargs, "semantic_max_wait_in_milliseconds", semantic_max_wait_in_milliseconds)


# 添加向量检索参数，统一 vector、hybrid 和 semantic_hybrid 的公共逻辑。
def _add_vector_options(
    search_kwargs: Dict[str, Any],
    *,
    vector_text: str,
    vector_fields: str,
    k: int,
    exhaustive: bool,
    weight: Optional[float],
    oversampling: Optional[float],
    filter_override: Optional[str],
    vector_similarity_threshold: Optional[float],
    search_score_threshold: Optional[float],
    vector_filter_mode: Optional[str],
) -> None:
    search_kwargs["vector_queries"] = _build_vector_query(
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
    _add_if_present(search_kwargs, "vector_filter_mode", vector_filter_mode)


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


# 提取 Agentic Retrieval 的主答案文本，保持服务端原始引用标记不变。
def _extract_agentic_answer_text(raw_response: Dict[str, Any]) -> str:
    try:
        return raw_response["response"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


# 收集答案中的唯一引用 ID，保持答案中首次出现的顺序。
def _extract_reference_ids(answer_text: str) -> List[str]:
    reference_ids: List[str] = []
    seen: set[str] = set()
    for match in re.findall(r"\[ref_id:(\d+)\]", answer_text):
        if match not in seen:
            reference_ids.append(match)
            seen.add(match)
    return reference_ids


# 读取 SDK 响应里的列表字段，避免 preview 字段返回 null 时影响 MCP 输出。
def _list_or_empty(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


# 建立 retrieval activity 到知识源名称的映射，补齐 citation 上下文。
def _build_activity_source_map(activities: List[Any]) -> Dict[Any, str]:
    activity_map: Dict[Any, str] = {}
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        activity_id = activity.get("id")
        knowledge_source_name = activity.get("knowledgeSourceName") or activity.get("knowledge_source_name")
        if activity_id is not None and knowledge_source_name:
            activity_map[activity_id] = knowledge_source_name
    return activity_map


# 提取 sourceData 中最常见的正文片段，兼容 Search Index 和 Work IQ 等来源。
def _extract_source_content(source_data: Any) -> Optional[str]:
    if not isinstance(source_data, dict):
        return None

    for key in ("content", "chunk", "text", "fabricAnswer"):
        value = source_data.get(key)
        if isinstance(value, str) and value:
            return value

    extracts = source_data.get("extracts")
    if isinstance(extracts, list):
        texts = [item.get("text") for item in extracts if isinstance(item, dict) and item.get("text")]
        if texts:
            return "\n\n".join(texts)

    return None


# 读取 sourceData 中常见标识字段，避免 Agent 再解析原始响应。
def _source_data_field(source_data: Any, *keys: str) -> Any:
    if not isinstance(source_data, dict):
        return None
    for key in keys:
        value = source_data.get(key)
        if value is not None:
            return value
    return None


# 归一化单条引用为 Agent 更容易消费的结构化 reference。
def _normalize_agentic_reference(ref: Dict[str, Any], activity_map: Dict[Any, str]) -> Dict[str, Any]:
    ref_id = ref.get("id")
    activity_source = ref.get("activitySource") or ref.get("activity_source")
    source_data = ref.get("sourceData", ref.get("source_data"))

    reference: Dict[str, Any] = {
        "ref_id": ref_id,
        "source_type": ref.get("type") or ref.get("source_type"),
        "title": ref.get("title"),
        "url": ref.get("url"),
        "knowledge_source_name": activity_map.get(activity_source),
        "activity_source": activity_source,
        "reranker_score": ref.get("rerankerScore", ref.get("reranker_score")),
        "content": _extract_source_content(source_data),
        "document_id": _source_data_field(source_data, "document_id", "documentId", "id") or ref.get("docKey"),
        "chunk_id": _source_data_field(source_data, "chunk_id", "chunkId", "chunkKey"),
        "doc_key": ref.get("docKey") or ref.get("doc_key"),
    }

    terms = _source_data_field(source_data, "terms")
    if terms is not None:
        reference["terms"] = terms

    return {key: value for key, value in reference.items() if value is not None}


# 默认只返回答案实际引用的证据；没有引用标记时退回返回全部证据。
def _select_referenced_sources(
    references: List[Dict[str, Any]],
    reference_ids: List[str],
) -> List[Dict[str, Any]]:
    if not reference_ids:
        return references

    by_id = {str(reference.get("ref_id")): reference for reference in references}
    return [by_id[ref_id] for ref_id in reference_ids if ref_id in by_id]


# 构造面向 Agent 的 Knowledge Base Retrieval 返回，不混入展示层格式。
def _build_agentic_response(
    raw_response: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    answer_text = _extract_agentic_answer_text(raw_response)
    activities = _list_or_empty(raw_response.get("activity"))
    raw_references = _list_or_empty(raw_response.get("references"))
    raw_metadata = raw_response.get("metadata")
    activity_map = _build_activity_source_map(activities)
    reference_ids = _extract_reference_ids(answer_text)
    normalized_references = [
        _normalize_agentic_reference(ref, activity_map)
        for ref in raw_references
        if isinstance(ref, dict)
    ]

    response: Dict[str, Any] = {
        "answer": {
            "text": answer_text,
            "used_ref_ids": reference_ids,
        },
        "references": _select_referenced_sources(normalized_references, reference_ids),
        "metadata": {
            **(raw_metadata if isinstance(raw_metadata, dict) else {}),
            **(metadata or {}),
            "referenced_count": len(reference_ids),
            "total_reference_count": len(normalized_references),
        },
    }

    if diagnostics:
        response["diagnostics"] = diagnostics

    return response


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
    advanced_options: str = "",
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
    Returns
    -------
    dict
        Response containing `documents`, `count`, optional `facets`, and `continuation_token`.
    """
    resolved_endpoint, key = _resolve_call_context()
    advanced = _parse_json_options(
        advanced_options,
        allowed_keys=COMMON_ADVANCED_OPTION_KEYS,
        label="advanced_options",
    )
    search_kwargs: Dict[str, Any] = {
        "top": top,
        "skip": skip,
        "include_total_count": True,
        "search_mode": search_mode,
    }
    _add_common_search_options(
        search_kwargs,
        select=_none_if_empty(select),
        filter=_none_if_empty(filter),
        search_fields=_none_if_empty(search_fields),
        debug=_none_if_empty(advanced.get("debug", "")),
    )

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
    query_caption: str = "extractive",
    query_caption_highlight_enabled: bool = True,
    query_answer: str = "",
    advanced_options: str = "",
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
    query_caption / query_answer: Optional[str]
        Semantic caption and answer modes (`"extractive"`, `"summary"`, etc.).

    Returns
    -------
    dict
        Response containing `documents`, `count`, optional `answers`, `captions`, and continuation metadata.
    """
    resolved_endpoint, key = _resolve_call_context()
    advanced = _parse_json_options(
        advanced_options,
        allowed_keys=COMMON_ADVANCED_OPTION_KEYS | SEMANTIC_ADVANCED_OPTION_KEYS,
        label="advanced_options",
    )

    search_kwargs: Dict[str, Any] = {
        "query_type": "semantic",
        "semantic_configuration_name": semantic_configuration,
        "top": top,
        "skip": skip,
        "include_total_count": True,
    }
    _add_common_search_options(
        search_kwargs,
        select=_none_if_empty(select),
        filter=_none_if_empty(filter),
        debug=_none_if_empty(advanced.get("debug", "")),
    )
    _add_semantic_options(
        search_kwargs,
        semantic_query=_none_if_empty(advanced.get("semantic_query", "")) or query,
        query_caption=query_caption,
        query_caption_highlight_enabled=query_caption_highlight_enabled,
        query_answer=_none_if_empty(query_answer),
        query_answer_count=_none_if_zero(advanced.get("query_answer_count", 0)),
        query_answer_threshold=_none_if_zero(advanced.get("query_answer_threshold", 0.0)),
        semantic_error_mode=_none_if_empty(advanced.get("semantic_error_mode", "")),
        semantic_max_wait_in_milliseconds=_none_if_zero(
            advanced.get("semantic_max_wait_in_milliseconds", 0)
        ),
    )

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
    select: str = "",
    filter: str = "",
    advanced_options: str = "",
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
    select / filter: Optional[str]
        Restrict fields returned or apply filters before vector scoring.

    Returns
    -------
    dict
        Response containing `documents`, `count`, and `continuation_token`.
    """
    resolved_endpoint, key = _resolve_call_context()
    advanced = _parse_json_options(
        advanced_options,
        allowed_keys=COMMON_ADVANCED_OPTION_KEYS | VECTOR_ADVANCED_OPTION_KEYS,
        label="advanced_options",
    )

    search_kwargs: Dict[str, Any] = {
        "top": k,
        "include_total_count": True,
    }
    _add_vector_options(
        search_kwargs,
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=advanced.get("exhaustive", False),
        weight=_none_if_zero(advanced.get("weight", 0.0)),
        oversampling=_none_if_zero(advanced.get("oversampling", 0.0)),
        filter_override=_none_if_empty(advanced.get("filter_override", "")),
        vector_similarity_threshold=_none_if_zero(advanced.get("vector_similarity_threshold", 0.0)),
        search_score_threshold=_none_if_zero(advanced.get("search_score_threshold", 0.0)),
        vector_filter_mode=_none_if_empty(advanced.get("vector_filter_mode", "")),
    )
    _add_common_search_options(
        search_kwargs,
        select=_none_if_empty(select),
        filter=_none_if_empty(filter),
        debug=_none_if_empty(advanced.get("debug", "")),
    )

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
    select: str = "",
    filter: str = "",
    search_fields: str = "",
    advanced_options: str = "",
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
    select / filter / search_fields: Optional[str]
        Customize returned fields, filters, or lexical scope.

    Returns
    -------
    dict
        Response containing merged `documents`, `count`, and continuation metadata.
    """
    resolved_endpoint, key = _resolve_call_context()
    advanced = _parse_json_options(
        advanced_options,
        allowed_keys=COMMON_ADVANCED_OPTION_KEYS | VECTOR_ADVANCED_OPTION_KEYS | HYBRID_ADVANCED_OPTION_KEYS,
        label="advanced_options",
    )
    hybrid_search_config = _build_hybrid_search(
        max_text_recall_size=_none_if_zero(advanced.get("max_text_recall_size", 0)),
        count_and_facet_mode=_none_if_empty(advanced.get("count_and_facet_mode", "")),
    )

    search_kwargs: Dict[str, Any] = {
        "top": top,
        "include_total_count": True,
    }
    _add_if_present(search_kwargs, "hybrid_search", hybrid_search_config)
    _add_vector_options(
        search_kwargs,
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=advanced.get("exhaustive", False),
        weight=_none_if_zero(advanced.get("weight", 0.0)),
        oversampling=_none_if_zero(advanced.get("oversampling", 0.0)),
        filter_override=_none_if_empty(advanced.get("filter_override", "")),
        vector_similarity_threshold=_none_if_zero(advanced.get("vector_similarity_threshold", 0.0)),
        search_score_threshold=_none_if_zero(advanced.get("search_score_threshold", 0.0)),
        vector_filter_mode=_none_if_empty(advanced.get("vector_filter_mode", "")),
    )
    _add_common_search_options(
        search_kwargs,
        select=_none_if_empty(select),
        filter=_none_if_empty(filter),
        search_fields=_none_if_empty(search_fields),
        debug=_none_if_empty(advanced.get("debug", "")),
    )

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
    select: str = "",
    filter: str = "",
    search_fields: str = "",
    query_caption: str = "extractive",
    query_caption_highlight_enabled: bool = True,
    query_answer: str = "",
    advanced_options: str = "",
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
    query_caption / query_answer: Optional[str]
        Semantic caption and answer modes (`"extractive"`, `"summary"`, etc.).
    select / filter / search_fields: Optional[str]
        Additional shaping of results and filters.

    Returns
    -------
    dict
        Response containing `documents`, `count`, optional `answers`, `captions`, and continuation metadata.
    """
    resolved_endpoint, key = _resolve_call_context()
    advanced = _parse_json_options(
        advanced_options,
        allowed_keys=(
            COMMON_ADVANCED_OPTION_KEYS
            | SEMANTIC_ADVANCED_OPTION_KEYS
            | VECTOR_ADVANCED_OPTION_KEYS
            | HYBRID_ADVANCED_OPTION_KEYS
        ),
        label="advanced_options",
    )
    hybrid_search_config = _build_hybrid_search(
        max_text_recall_size=_none_if_zero(advanced.get("max_text_recall_size", 0)),
        count_and_facet_mode=_none_if_empty(advanced.get("count_and_facet_mode", "")),
    )

    search_kwargs: Dict[str, Any] = {
        "query_type": "semantic",
        "semantic_configuration_name": semantic_configuration,
        "top": top,
        "include_total_count": True,
    }
    _add_if_present(search_kwargs, "hybrid_search", hybrid_search_config)
    _add_vector_options(
        search_kwargs,
        vector_text=vector_text,
        vector_fields=vector_fields,
        k=k,
        exhaustive=advanced.get("exhaustive", False),
        weight=_none_if_zero(advanced.get("weight", 0.0)),
        oversampling=_none_if_zero(advanced.get("oversampling", 0.0)),
        filter_override=_none_if_empty(advanced.get("filter_override", "")),
        vector_similarity_threshold=_none_if_zero(advanced.get("vector_similarity_threshold", 0.0)),
        search_score_threshold=_none_if_zero(advanced.get("search_score_threshold", 0.0)),
        vector_filter_mode=_none_if_empty(advanced.get("vector_filter_mode", "")),
    )
    _add_common_search_options(
        search_kwargs,
        select=_none_if_empty(select),
        filter=_none_if_empty(filter),
        search_fields=_none_if_empty(search_fields),
        debug=_none_if_empty(advanced.get("debug", "")),
    )
    _add_semantic_options(
        search_kwargs,
        semantic_query=_none_if_empty(advanced.get("semantic_query", "")),
        query_caption=query_caption,
        query_caption_highlight_enabled=query_caption_highlight_enabled,
        query_answer=_none_if_empty(query_answer),
        query_answer_count=_none_if_zero(advanced.get("query_answer_count", 0)),
        query_answer_threshold=_none_if_zero(advanced.get("query_answer_threshold", 0.0)),
        semantic_error_mode=_none_if_empty(advanced.get("semantic_error_mode", "")),
        semantic_max_wait_in_milliseconds=_none_if_zero(
            advanced.get("semantic_max_wait_in_milliseconds", 0)
        ),
    )

    return await _execute_search(
        endpoint=resolved_endpoint,
        key=key,
        index_name=index_name,
        search_text=query,
        search_kwargs=search_kwargs,
    )


# 校验知识源配置的最小结构，避免把明显错误的配置交给 SDK。
def _validate_knowledge_source_configs(sources: List[Dict[str, Any]]) -> None:
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each knowledge source config must be an object.")
        if "knowledgeSourceName" not in source:
            raise ValueError(f"Missing required 'knowledgeSourceName' in source config: {source}")
        if "kind" not in source:
            raise ValueError(f"Missing required 'kind' in source config: {source}")


# 解析客户传入的知识源 JSON 配置，支持单个对象或对象数组。
def _parse_knowledge_source_configs(config_str: str) -> List[Dict[str, Any]]:
    stripped = config_str.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("knowledge_source_configs must be a JSON object or JSON array.") from exc

    sources = [parsed] if isinstance(parsed, dict) else parsed
    if not isinstance(sources, list):
        raise ValueError("knowledge_source_configs must be a JSON object or JSON array.")
    _validate_knowledge_source_configs(sources)
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
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    """调用 Azure AI Search Knowledge Base Retrieval SDK。"""
    intent_query = _none_if_empty(intent_query)
    reasoning_effort = _none_if_empty(reasoning_effort)
    output_mode = _none_if_empty(output_mode)
    max_runtime_seconds = _none_if_zero(max_runtime_seconds)
    max_output_size = _none_if_zero(max_output_size)
    max_output_documents = _none_if_zero(max_output_documents)
    knowledge_source_configs = _none_if_empty(knowledge_source_configs)
    query_source_authorization = _none_if_empty(query_source_authorization)

    resolved_endpoint, key = _resolve_call_context(admin=True)

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
                role="user",
                content=[KnowledgeBaseMessageTextContent(text=query)],
            )
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
            request_kwargs["knowledge_source_params"] = _parse_knowledge_source_configs(knowledge_source_configs)
        except ValueError as exc:
            raise ValueError(f"Failed to parse knowledge_source_configs: {exc}") from exc

    request = KnowledgeBaseRetrievalRequest(**request_kwargs)

    timeout_budget: int = _build_agentic_timeout_budget(max_runtime_seconds)
    client = KnowledgeBaseRetrievalClient(
        endpoint=resolved_endpoint,
        credential=AzureKeyCredential(key),
        knowledge_base_name=knowledge_base_name,
    )

    started_at = time.perf_counter()
    try:
        async with client:
            retrieve_kwargs: Dict[str, Any] = {
                "retrieval_request": request,
                "timeout": timeout_budget,
            }
            if query_source_authorization:
                retrieve_kwargs["query_source_authorization"] = query_source_authorization
            result = await asyncio.wait_for(
                client.retrieve(**retrieve_kwargs),
                timeout=timeout_budget,
            )
        raw_data = _to_plain_data(result)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        response_metadata = {
            "knowledge_base_name": knowledge_base_name,
            "output_mode": output_mode,
            "reasoning_effort": reasoning_effort,
            "elapsed_ms": elapsed_ms,
        }
        diagnostics = None
        if include_diagnostics:
            diagnostics = {
                "request": request.as_dict(),
                "timeout_budget_seconds": timeout_budget,
                "response": _list_or_empty(raw_data.get("response")),
                "raw_references": _list_or_empty(raw_data.get("references")),
                "activity": _list_or_empty(raw_data.get("activity")),
            }
        return _build_agentic_response(raw_data, metadata=response_metadata, diagnostics=diagnostics)
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
            "but search tools require AZURE_SEARCH_ENDPOINT or the --endpoint startup argument."
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
