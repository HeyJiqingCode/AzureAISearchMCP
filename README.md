# Azure AI Search MCP Server

A custom Model Context Protocol (MCP) server for exposing Azure AI Search retrieval capabilities to MCP-compatible clients.

This server provides one customer-friendly tool layer for keyword search, semantic search, vector search, hybrid search, semantic hybrid search, and Azure AI Search Knowledge Base retrieval.

## When To Use This

Use this project when you want one MCP server that exposes multiple Azure AI Search retrieval modes behind a small, stable set of tools.

Azure AI Search Knowledge Bases also expose a native MCP endpoint for direct Knowledge Base retrieval:

```text
https://<your-service-name>.search.windows.net/knowledgebases/<your-knowledge-base-name>/mcp?api-version=<api-version>
```

This native endpoint is available for Knowledge Base objects, not ordinary search indexes. Microsoft documents it in [Query a knowledge base using the retrieve action or MCP endpoint](https://learn.microsoft.com/azure/search/agentic-retrieval-how-to-retrieve#call-the-mcp-endpoint).

If your client only needs Knowledge Base retrieval, the native endpoint can be enough. This project is useful when you also want direct access to traditional search modes such as keyword, semantic, vector, hybrid, and semantic hybrid retrieval.

## Features

- Keyword search with filters, selected fields, and search field scoping.
- Semantic search with semantic ranking, captions, and answers.
- Vector search using integrated vectorization from an index vectorizer.
- Hybrid search combining keyword and vector retrieval.
- Semantic hybrid search with hybrid retrieval plus semantic reranking.
- Knowledge Base retrieval through the Azure AI Search Python SDK preview client.
- Focused tool parameters for common use, with low-frequency SDK tuning under `advanced_options`.

## Quick Start

### 1. Install

```bash
git clone https://github.com/HeyJiqingCode/AzureAISearchMCP.git
cd AzureAISearchMCP

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file:

```bash
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_QUERY_KEY=your-query-key
AZURE_SEARCH_ADMIN_KEY=your-admin-key

# Optional
# MCP_HOST=0.0.0.0
# MCP_PORT=8000
# AZURE_SEARCH_TIMEOUT=30
# AZURE_SEARCH_AGENTIC_TIMEOUT=90
# AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER=30
```

Traditional search tools use `AZURE_SEARCH_QUERY_KEY`. The `agentic_retrieval` tool uses `AZURE_SEARCH_ADMIN_KEY`.

### 3. Run

For local MCP clients using stdio:

```bash
python src/mcp/server.py
```

For Streamable HTTP:

```bash
python src/mcp/server.py --transport streamable-http --host 0.0.0.0 --port 8000
```

For SSE:

```bash
python src/mcp/server.py --transport sse --host 0.0.0.0 --port 8000
```

You can set `AZURE_SEARCH_ENDPOINT` in the environment or pass `--endpoint` when starting the server.

### 4. Connect an MCP Client

Example Streamable HTTP configuration:

```json
{
  "mcpServers": {
    "AzureAISearch": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

If the server is deployed behind a gateway or proxy that requires authentication, configure the required headers in your MCP client.

## Configuration

| Setting | Required | Used by | Description |
| --- | --- | --- | --- |
| `AZURE_SEARCH_ENDPOINT` | Yes | All tools | Azure AI Search service endpoint. |
| `AZURE_SEARCH_QUERY_KEY` | Yes | Search tools | Query key for keyword, semantic, vector, hybrid, and semantic hybrid tools. |
| `AZURE_SEARCH_ADMIN_KEY` | For Agentic | `agentic_retrieval` | Admin key used by Knowledge Base retrieval. Treat it as sensitive. |
| `MCP_HOST` | No | HTTP transports | Host used by `http`, `streamable-http`, and `sse`. Default: `0.0.0.0`. |
| `MCP_PORT` | No | HTTP transports | Port used by `http`, `streamable-http`, and `sse`. Default: `8000`. |
| `AZURE_SEARCH_TIMEOUT` | No | Search tools | How long this MCP server waits for standard search calls. Default: `30`. |
| `AZURE_SEARCH_AGENTIC_TIMEOUT` | No | `agentic_retrieval` | How long this MCP server waits for Agentic Retrieval calls. Default: `90`. |
| `AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER` | No | `agentic_retrieval` | Extra wait time added when `max_runtime_seconds` is set. Default: `30`. |

### Timeout Settings

Standard search tools use `AZURE_SEARCH_TIMEOUT`.

Agentic Retrieval uses two timeout layers:

- `max_runtime_seconds`: a tool argument that asks Azure AI Search to stop the Knowledge Base retrieve operation after that many seconds.
- `AZURE_SEARCH_AGENTIC_TIMEOUT`: how long this MCP server is willing to wait for Azure AI Search to respond.
- `AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER`: extra waiting time added when `max_runtime_seconds` is set.

Example: if `max_runtime_seconds=60` and `AZURE_SEARCH_AGENTIC_TIMEOUT_BUFFER=30`, this MCP server waits up to 90 seconds. This gives Azure AI Search 60 seconds to run, plus 30 seconds for network and response overhead.

## Tools

| Tool | Use when |
| --- | --- |
| `simple_search` | You need standard keyword or BM25 search. |
| `semantic_search` | You need semantic ranking, captions, or answers from a semantic configuration. |
| `vector_search` | You need vector-only similarity search using integrated vectorization. |
| `hybrid_search` | You need keyword and vector retrieval in one query. |
| `semantic_hybrid_search` | You need hybrid retrieval plus semantic reranking, captions, or answers. |
| `agentic_retrieval` | You need Azure AI Search Knowledge Base / Agentic Retrieval. |

Search tool responses include:

- `documents`: normalized search results, including fields such as `@search.score` when returned.
- `count`: total document count when available.
- `answers`, `captions`, `facets`: included when returned by Azure AI Search.
- `continuation_token`: included when the SDK reports a continuation token.

## Tool Reference

### `simple_search`

Keyword search over an index using simple query syntax.

```text
simple_search(index_name, query, top=5, skip=0, search_fields="", select="", filter="", search_mode="any", advanced_options="")
```

```json
{
  "tool": "simple_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to configure Wi-Fi for Windows PC?",
    "top": 3,
    "select": "title,body"
  }
}
```

### `semantic_search`

Semantic reranked search for indexes with semantic configuration enabled.

```text
semantic_search(index_name, query, semantic_configuration, top=5, skip=0, select="", filter="", query_caption="extractive", query_caption_highlight_enabled=True, query_answer="", advanced_options="")
```

```json
{
  "tool": "semantic_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to configure Wi-Fi for Windows PC?",
    "semantic_configuration": "default",
    "top": 3
  }
}
```

### `vector_search`

Vector-only similarity search using integrated vectorization.

```text
vector_search(index_name, vector_fields, vector_text, k=10, select="", filter="", advanced_options="")
```

```json
{
  "tool": "vector_search",
  "arguments": {
    "index_name": "knowledge-base",
    "vector_fields": "text_vector",
    "vector_text": "How to configure Wi-Fi for Windows PC?",
    "k": 5,
    "select": "title,summary"
  }
}
```

### `hybrid_search`

Hybrid keyword and vector search using Reciprocal Rank Fusion.

```text
hybrid_search(index_name, query, vector_fields, vector_text, k=10, top=10, select="", filter="", search_fields="", advanced_options="")
```

```json
{
  "tool": "hybrid_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to configure Wi-Fi for Windows PC?",
    "vector_fields": "text_vector",
    "vector_text": "How to configure Wi-Fi for Windows PC?",
    "k": 20,
    "top": 5,
    "search_fields": "title,body"
  }
}
```

### `semantic_hybrid_search`

Hybrid retrieval with semantic reranking, captions, and answers.

```text
semantic_hybrid_search(index_name, query, vector_fields, semantic_configuration, vector_text, k=50, top=10, select="", filter="", search_fields="", query_caption="extractive", query_caption_highlight_enabled=True, query_answer="", advanced_options="")
```

```json
{
  "tool": "semantic_hybrid_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to configure Wi-Fi for Windows PC?",
    "vector_fields": "text_vector",
    "semantic_configuration": "default",
    "vector_text": "How to configure Wi-Fi for Windows PC?",
    "k": 30,
    "top": 5,
    "query_caption": "extractive",
    "query_answer": "extractive"
  }
}
```

### `advanced_options`

`advanced_options` is a JSON object string for less common SDK tuning settings. Unsupported keys are rejected.

Common keys:

```text
debug
```

Semantic keys:

```text
semantic_query, query_answer_count, query_answer_threshold, semantic_error_mode, semantic_max_wait_in_milliseconds
```

Vector keys:

```text
exhaustive, weight, oversampling, filter_override, vector_filter_mode, vector_similarity_threshold, search_score_threshold
```

Hybrid keys:

```text
max_text_recall_size, count_and_facet_mode
```

Example:

```json
{
  "advanced_options": "{\"max_text_recall_size\":100,\"vector_filter_mode\":\"preFilter\"}"
}
```

## Agentic Retrieval

`agentic_retrieval` runs Azure AI Search Knowledge Base retrieval through the Python SDK preview client.

```text
agentic_retrieval(knowledge_base_name, query, intent_query="", reasoning_effort="low", output_mode="answerSynthesis", include_activity=True, max_runtime_seconds=0, max_output_size=0, max_output_documents=0, knowledge_source_configs="", query_source_authorization="", include_diagnostics=False)
```

Frequently used arguments:

| Argument | Description |
| --- | --- |
| `knowledge_base_name` | Azure AI Search Knowledge Base name. |
| `query` | User question or retrieval query. |
| `reasoning_effort` | `minimal`, `low`, or `medium`. Default: `low`. |
| `output_mode` | `answerSynthesis` or `extractedData`. Default: `answerSynthesis`. |
| `include_activity` | Include query planning and retrieval activity details. |
| `max_runtime_seconds` | Ask Azure AI Search to cap service-side retrieval runtime. |
| `max_output_size` | Bound the grounded response payload size. |
| `max_output_documents` | Cap final grounding document count. |
| `knowledge_source_configs` | JSON object or JSON array string for runtime knowledge source settings. |
| `query_source_authorization` | End-user token for query-time permission enforcement. |
| `include_diagnostics` | Include normalized request details and timeout budget for troubleshooting. Default: `false`. |

`answerSynthesis` requires message-based retrieval, so use `reasoning_effort="low"` or `reasoning_effort="medium"`. Use `reasoning_effort="minimal"` with `output_mode="extractedData"` for direct semantic intent retrieval.

Response is structured for agent consumption:

- `answer`: object containing `text` and `citation_markers`. The answer text keeps Azure AI Search citation markers such as `[ref_id:0]`.
- `references`: normalized evidence objects referenced by `answer.text`, with fields such as `ref_id`, `source_type`, `title`, `url`, `content`, `document_id`, `chunk_id`, `doc_key`, `knowledge_source_name`, `activity_source`, and `reranker_score`.
- `metadata`: request and retrieval metadata such as `knowledge_base_name`, `output_mode`, `reasoning_effort`, `elapsed_ms`, `referenced_count`, and `total_reference_count`.
- `diagnostics`: included only when `include_diagnostics=true`; contains the normalized SDK request, timeout budget, raw response, raw references, and activity details.

To receive source chunks in `references[].content`, set `includeReferenceSourceData=true` in the relevant `knowledge_source_configs` entry. Azure AI Search only returns `sourceData` when the knowledge source is configured to include it.

### Basic Example

```json
{
  "tool": "agentic_retrieval",
  "arguments": {
    "knowledge_base_name": "kb-support",
    "query": "How do I reset my VPN password?",
    "reasoning_effort": "low",
    "output_mode": "answerSynthesis",
    "include_activity": true
  }
}
```

### Minimal Intent Example

```json
{
  "tool": "agentic_retrieval",
  "arguments": {
    "knowledge_base_name": "kb-support",
    "query": "How do I reset my VPN password?",
    "reasoning_effort": "minimal",
    "output_mode": "extractedData"
  }
}
```

### Knowledge Source Configuration

Use `knowledge_source_configs` to specify one or more knowledge sources with per-source settings. Pass a JSON object or JSON array encoded as a string.

Common knowledge source config keys:

| Key | Type | Description |
| --- | --- | --- |
| `knowledgeSourceName` | string | Knowledge source name. |
| `kind` | string | Source type, such as `searchIndex`, `web`, `azureBlob`, `indexedOneLake`, or another preview kind supported by the SDK. |
| `includeReferences` | bool | Include document references. |
| `includeReferenceSourceData` | bool | Include source data in references. |
| `rerankerThreshold` | float | Minimum reranker score threshold. |
| `alwaysQuerySource` | bool | Force querying when supported by the selected source kind. |
| `failOnError` | bool | Treat this source as required. |
| `maxOutputDocuments` | int | Cap candidate documents from this source. |
| `filterAddOn` | string | Runtime OData filter for search index knowledge sources. |
| `count`, `freshness`, `language`, `market` | mixed | Web source controls. |
| `filterExpressionAddOn` | string | KQL filter expression for SharePoint-style sources. |

Single source:

```json
{
  "tool": "agentic_retrieval",
  "arguments": {
    "knowledge_base_name": "kb-support",
    "query": "How do I reset my VPN password?",
    "knowledge_source_configs": "{\"knowledgeSourceName\":\"ks-docs\",\"kind\":\"searchIndex\",\"includeReferences\":true}"
  }
}
```

Multiple sources:

```json
{
  "tool": "agentic_retrieval",
  "arguments": {
    "knowledge_base_name": "kb-support",
    "query": "Latest security updates",
    "knowledge_source_configs": "[{\"knowledgeSourceName\":\"ks-docs\",\"kind\":\"searchIndex\",\"includeReferences\":true},{\"knowledgeSourceName\":\"ks-web\",\"kind\":\"web\",\"count\":10,\"freshness\":\"week\"}]"
  }
}
```

Search field selection is handled by Azure AI Search based on Knowledge Base and index configuration.

## Docker

Build the image:

```bash
docker build -t azure-ai-search-mcp:2.0.0 -f Dockerfile .
```

Run the container:

```bash
docker run -itd -p 8000:8000 --name AzureAISearch \
  -e AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net \
  -e AZURE_SEARCH_QUERY_KEY=your-query-key \
  -e AZURE_SEARCH_ADMIN_KEY=your-admin-key \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  azure-ai-search-mcp:2.0.0
```

## More Details

See [MCP Server for Azure AI Search](https://heyjiqing.notion.site/MCP-Server-for-Azure-AI-Search-294de7b6e4e8805faccad1f60cc255e2?pvs=74)
