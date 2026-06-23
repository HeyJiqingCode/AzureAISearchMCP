# Azure AI Search MCP Server

A Model Context Protocol (MCP) server that exposes Azure AI Search capabilities across multiple retrieval modes (keyword, semantic, vector, hybrid, plus Knowledge Base retrieval through the latest Python SDK preview APIs).

## Features
- Execute keyword searches using simple syntax.
- Run semantic queries with optional captions and answers.
- Perform pure vector similarity search.
- Combine keyword and vector retrieval (hybrid search).
- Apply semantic reranking on hybrid results for richer answers.
- Call Azure AI Search Knowledge Base retrieval with `azure-search-documents` 12.1 preview SDK support.
- Tune modern vector and hybrid retrieval settings such as oversampling, vector filter mode, filter overrides, semantic debug, and `hybridSearch.maxTextRecallSize`.
- Support integrated vectorization when your index has an attached vectorizer (no manual embeddings required).
- Tools read `AZURE_SEARCH_ENDPOINT` and key env vars at runtime. Traditional search tools fall back to `AZURE_SEARCH_QUERY_KEY`, while the agentic tool falls back to `AZURE_SEARCH_ADMIN_KEY`. Pass `endpoint`/`api_key` parameters only when you need to override per-call.

## Quick Start

### Local Development

**1/ Clone Repo**
```bash
git clone https://github.com/HeyJiqingCode/AzureAISearchMCP.git
cd AzureAISearchMCP
```

**2/ Install dependencies**:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3/ Create a `.env` file**
```bash
# Azure AI Search Endpoint
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net

# Azure AI Search Keys (QueryKey AdminKey)
AZURE_SEARCH_QUERY_KEY=your-query-key
AZURE_SEARCH_ADMIN_KEY=your-admin-key   # required for agentic_retrieval

# Server and timeout settings (Optional)
# MCP_HOST=0.0.0.0
# MCP_PORT=8000
# AZURE_SEARCH_TIMEOUT=120
```

**4/ Run the server**:
```bash
# For stdio transport (default)
python src/mcp/server.py

# For SSE transport
python src/mcp/server.py --transport sse --host 0.0.0.0 --port 8000

# For Streamable HTTP transport
python src/mcp/server.py --transport streamable-http --host 0.0.0.0 --port 8000
```

The server can start without `AZURE_SEARCH_ENDPOINT` if your MCP client passes `endpoint` in each tool call.

**5/ Add MCP Server to your client for Streamable HTTP**
```json
{
  "mcpServers": {
    "AzureAISearch": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {
         "Content-Type": "application/json",
         "Authorization": "Bearer your_token"
      }
    }
  }
}
```

### Docker

**1/ Clone Repo**
```bash
git clone https://github.com/HeyJiqingCode/AzureAISearchMCP.git
cd AzureAISearchMCP
```

**2/ Build Docker Image**
```bash
docker build -t azure-ai-search-mcp:2.0.0 -f Dockerfile .
```

**3/ Run the container**:
```bash
docker run -itd -p 8000:8000 --name AzureAISearch \
  -e AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net \
  -e AZURE_SEARCH_QUERY_KEY=your-query-key \
  -e AZURE_SEARCH_ADMIN_KEY=your-admin-key \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  azure-ai-search-mcp:2.0.0
```

**4/ Add MCP Server for HTTP transport**
```json
{
  "mcpServers": {
    "AzureAISearch": {
      "type": "streamableHttp",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": {
         "Content-Type": "application/json",
         "Authorization": "Bearer your_token"
      }
    }
  }
}
```

## Available Tools

Each tool accepts an optional `api_key` and `endpoint` so you can override defaults at invocation time. All responses include:
- `documents`: list of normalized documents (with `@search.score`, etc.).
- `count`: total number of documents matched (if available).
- `answers`, `captions`, `facets`: when returned by the service.
- `continuation_token`: set if further paging is available.

### `simple_search`

Keyword (BM25) search over an index using simple query syntax, with optional filters and field selection.

**Parameters:**

index_name, query, top=5, skip=0, search_fields=None, select=None, filter=None, search_mode="any", api_key=None, endpoint=None

**Example Usage:**
```json
{
  "tool": "simple_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to config Wifi for Windows PC?",
    "top": 3,
    "select": "title,body"
  }
}
```

### `semantic_search`

Semantic reranked search returning optional captions and answers when the index has semantic configuration enabled.

**Parameters:**

index_name, query, semantic_configuration, top=5, skip=0, select=None, filter=None, semantic_query=None, query_caption="extractive", query_caption_highlight_enabled=True, query_answer=None, semantic_error_mode=None, semantic_max_wait_in_milliseconds=None, debug=None, api_key=None, endpoint=None

**Example Usage:**

```json
{
  "tool": "semantic_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to config Wifi for Windows PC?",
    "semantic_configuration": "default",
    "top": 3
  }
}
```

### `vector_search`

Vector-only similarity search using integrated vectorization (text-to-embedding) over specified vector fields.

**Parameters:**

index_name, vector_fields, vector_text, k=10, exhaustive=False, weight=None, oversampling=None, filter_override=None, vector_filter_mode=None, vector_similarity_threshold=None, search_score_threshold=None, select=None, filter=None, debug=None, api_key=None, endpoint=None

**Example Usage:**

```json
{
  "tool": "vector_search",
  "arguments": {
    "index_name": "knowledge-base",
    "vector_fields": "text_vector",
    "vector_text": "How to config Wifi for Windows PC?",
    "k": 5,
    "select": "title,summary"
  }
}
```

### `hybrid_search`

Hybrid (keyword + vector) search that fuses BM25 and vector similarity results using Reciprocal Rank Fusion.

**Parameters:**

index_name, query, vector_fields, vector_text, k=10, top=10, exhaustive=False, weight=None, oversampling=None, filter_override=None, vector_filter_mode=None, vector_similarity_threshold=None, search_score_threshold=None, max_text_recall_size=None, count_and_facet_mode=None, select=None, filter=None, search_fields=None, debug=None, api_key=None, endpoint=None

**Example Usage:**

```json
{
  "tool": "hybrid_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to config Wifi for Windows PC?",
    "vector_fields": "text_vector",
    "vector_text": "How to config Wifi for Windows PC?",
    "k": 20,
    "top": 5,
    "search_fields": "title,body"
  }
}
```

### `semantic_hybrid_search`

Hybrid (keyword + vector) search with semantic reranking, captions, and answers when configured.

**Parameters:**

index_name, query, vector_fields, semantic_configuration, vector_text, k=50, top=10, exhaustive=False, weight=None, oversampling=None, filter_override=None, vector_filter_mode=None, vector_similarity_threshold=None, search_score_threshold=None, max_text_recall_size=None, count_and_facet_mode=None, select=None, filter=None, search_fields=None, semantic_query=None, query_caption="extractive", query_caption_highlight_enabled=True, query_answer=None, semantic_error_mode=None, semantic_max_wait_in_milliseconds=None, debug=None, api_key=None, endpoint=None

**Example Usage:**

```json
{
  "tool": "semantic_hybrid_search",
  "arguments": {
    "index_name": "knowledge-base",
    "query": "How to config Wifi for Windows PC?",
    "vector_fields": "text_vector",
    "semantic_configuration": "default",
    "vector_text": "How to config Wifi for Windows PC?",
    "k": 30,
    "top": 5,
    "query_caption": "extractive",
    "query_answer": "extractive"
  }
}
```

### `agentic_retrieval`

Run Azure AI Search Knowledge Base retrieval through the Python SDK preview client. Requires `AZURE_SEARCH_ADMIN_KEY` or an admin key passed via `api_key`.

**Parameters (frequently used):**

- `knowledge_base_name` (str, required)
- `query` (str, required)
- `intent_query` (Optional[str]) – direct semantic intent; this uses intent-based retrieval
- `reasoning_effort` (str) – `minimal`, `low`, or `medium`; default is `low`
- `output_mode` (str) – `answerSynthesis` or `extractedData`; default is `answerSynthesis`
- `include_activity` (bool)
- `max_runtime_seconds`, `max_output_size`, `max_output_documents` (Optional[int])
- `knowledge_source_configs` (Optional[str]) – JSON string for configuring one or more knowledge sources. The older `key=value` format is still accepted for compatibility.
- `query_source_authorization` (Optional[str]) – end-user token for query-time permission enforcement
- `api_key`, `endpoint`

`answerSynthesis` requires message-based retrieval (`low` or `medium`). Use `reasoning_effort="minimal"` with `output_mode="extractedData"` for direct semantic intent retrieval.

**Knowledge Source Configuration:**

Use `knowledge_source_configs` to specify one or more knowledge sources with per-source settings. Prefer a JSON object or JSON array encoded as a string.

**Supported Parameters by Source Type:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `knowledgeSourceName` | string | Knowledge source name |
| `kind` | string | Source type, such as `searchIndex`, `web`, `azureBlob`, `indexedOneLake`, or preview kinds supported by the SDK |
| `includeReferences` | bool | Include document references |
| `includeReferenceSourceData` | bool | Include source data in references |
| `rerankerThreshold` | float | Minimum reranker score threshold |
| `alwaysQuerySource` | bool | Force querying when supported by the selected source kind |
| `failOnError` | bool | Treat this source as required |
| `maxOutputDocuments` | int | Cap candidate documents from this source |
| `filterAddOn` | string | Runtime OData filter for search index knowledge sources |
| `count`, `freshness`, `language`, `market` | mixed | Web source controls |
| `filterExpressionAddOn` | string | KQL filter expression for SharePoint-style sources |

**Format Rules:**
- Preferred: JSON object for one source, or JSON array for multiple sources
- Compatibility: `knowledgeSourceName=ks-docs, kind=searchIndex, includeReferences=true`
- **Note**: Search field selection is handled automatically by the API based on index configuration

**Example Usage:**

```json
{
  "tool": "agentic_retrieval",
  "arguments": {
    "knowledge_base_name": "kb-support",
    "query": "How do I reset my VPN password?",
    "reasoning_effort": "low",
    "output_mode": "answerSynthesis",
    "include_activity": true,
    "knowledge_source_configs": "{\"knowledgeSourceName\":\"ks-docs\",\"kind\":\"searchIndex\",\"includeReferences\":true}"
  }
}
```

**Minimal intent example:**

```json
{
  "tool": "agentic_retrieval",
  "arguments": {
    "knowledge_base_name": "kb-support",
    "query": "How do I reset my VPN password?",
    "reasoning_effort": "minimal",
    "output_mode": "extractedData",
    "knowledge_source_configs": "{\"knowledgeSourceName\":\"ks-docs\",\"kind\":\"searchIndex\",\"includeReferences\":true}"
  }
}
```

**Multiple sources example:**

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

Response includes a convenience `answer`, formatted `references`, the SDK response fields (`response`, `activity`, `raw_references`, `metadata`), and the normalized SDK request body used for the call.

## More Details

See [MCP Server for Azure AI Search](https://heyjiqing.notion.site/MCP-Server-for-Azure-AI-Search-294de7b6e4e8805faccad1f60cc255e2?pvs=74)
