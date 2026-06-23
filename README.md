# Azure AI Search MCP Server

A Model Context Protocol (MCP) server that exposes Azure AI Search capabilities across multiple retrieval modes (keyword, semantic, vector, hybrid, plus the new agentic retrieval preview APIs).

## Features
- Execute keyword searches using simple syntax.
- Run semantic queries with optional captions and answers.
- Perform pure vector similarity search.
- Combine keyword and vector retrieval (hybrid search).
- Apply semantic reranking on hybrid results for richer answers.
- Call Azure AI Search agentic retrieval (knowledge base multi-query pipeline, REST `2025-11-01-preview`).
- Support integrated vectorization when your index has an attached vectorizer (no manual embeddings required).
- Tools read `AZURE_SEARCH_ENDPOINT` and key env vars at runtime. Traditional search tools fall back to `AZURE_SEARCH_QUERY_KEY`, while the agentic tool falls back to `AZURE_SEARCH_ADMIN_KEY`. Pass `endpoint`/`api_key` parameters only when you need to override per-call.

## Quick Start

### Local Development

**1/ Clone Repo**
```bash
git clone https://github.com/HeyJiqingCode/mcp.git
```

**2/ Install dependencies**:
```bash
cd mcp/ai-search/
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

# Timeout (Optional)
# AZURE_SEARCH_TIMEOUT=120
```

**4/ Run the server**:
```bash
# For stdio transport (default)
python src/mcp/server.py

# For SSE transport
python src/mcp/server.py --transport sse

# For streamable-http transport  
python src/mcp/server.py --transport streamable-http
```

**5/ Add MCP Server to your client**
```json
{
  "mcpServers": {
    "AIFoundryAgent": {
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
git clone https://github.com/HeyJiqingCode/mcp.git
```

**2/ Build Docker Image**
```bash
cd mcp/ai-search/
docker build -t azure-ai-search-mcp:1.1.0 -f Dockerfile .
```

**3/ Run the container**:
```bash
docker run -itd -p 8000:8000 --name AzureAISearch \
  -e AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net \
  -e AZURE_SEARCH_QUERY_KEY=your-query-key \
  -e AZURE_SEARCH_ADMIN_KEY=your-admin-key \
  azure-ai-search-mcp:1.1.0
```

**4/ Add MCP Server for HTTP transport**
```json
{
  "mcpServers": {
    "AIFoundryAgent": {
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

**Paremeters:**

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

**Paremeters:**

index_name, query, semantic_configuration, top=5, skip=0, captions="extractive", answers=None, select=None, filter=None, api_key=None, endpoint=None

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

**Paremeters:**

index_name, vector_fields, vector_text, k=10, exhaustive=False, weight=None, select=None, filter=None, api_key=None, endpoint=None

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

**Paremeters:**

index_name, query, vector_fields, vector_text, k=10, top=10, exhaustive=False, weight=None, select=None, filter=None, search_fields=None, api_key=None, endpoint=None

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

**Paremeters:**

index_name, query, vector_fields, semantic_configuration, vector_text, k=50, top=10, exhaustive=False, weight=None, captions="extractive", answers=None, select=None, filter=None, search_fields=None, api_key=None, endpoint=None

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

Run the Azure AI Search agentic retrieval pipeline (knowledge base multi-query orchestration, preview). Requires `AZURE_SEARCH_ADMIN_KEY` or an admin key passed via `api_key`.

**Parameters (frequently used):**

- `knowledge_base_name` (str, required)
- `query` (str, required)
- `intent_query` (Optional[str])
- `reasoning_effort` (Optional[str]) – `minimal`, `low`, or `medium`
- `output_mode` (str) – `answerSynthesis` or `extractiveData`
- `include_activity` (bool)
- `max_runtime_seconds`, `max_output_size` (Optional[int])
- `knowledge_source_configs` (Optional[str]) – Key-value format for configuring knowledge sources. Format: `"knowledgeSourceName=name, kind=type, param=value; knowledgeSourceName=name2, ..."`. Each source can be independently configured with type-specific parameters.
- `api_key`, `endpoint`

**Knowledge Source Configuration:**

Use `knowledge_source_configs` to specify one or more knowledge sources with per-source settings. Parameters can be in **any order** - the parser is order-independent.

**Supported Parameters by Source Type:**

| Parameter | Type | searchIndex | web | remoteSharePoint | Description |
|-----------|------|-------------|-----|------------------|-------------|
| `knowledgeSourceName` | string | ✅ Required | ✅ Required | ✅ Required | Knowledge source name |
| `kind` | string | ✅ Required | ✅ Required | ✅ Required | Source type |
| `includeReferences` | bool | ✅ | ✅ | ✅ | Include document references |
| `alwaysQuerySource` | bool | ✅ | ✅ | ✅ | Force querying even if not needed |
| `rerankerThreshold` | float | ✅ | ✅ | ✅ | Minimum reranker score threshold |
| `includeReferenceSourceData` | bool | ✅ | ✅ | ✅ | Include source data in references |
| `filterAddOn` | string | ✅ | ❌ | ❌ | OData filter expression |
| `count` | int | ❌ | ✅ | ❌ | Number of results |
| `freshness` | string | ❌ | ✅ | ❌ | Result freshness (day/week/month) |
| `language` | string | ❌ | ✅ | ❌ | Result language (e.g., zh-CN) |
| `market` | string | ❌ | ✅ | ❌ | Result market (e.g., zh-CN) |
| `filterExpressionAddOn` | string | ❌ | ❌ | ✅ | KQL filter expression |

**Format Rules:**
- Parameters can be specified in **any order**
- Use `,` to separate key-value pairs within a source
- Use `;` to separate multiple sources
- Boolean values: `true` or `false` (lowercase)
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
    "knowledge_source_configs": "knowledgeSourceName=ks-docs, kind=searchIndex, includeReferences=true"
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
    "knowledge_source_configs": "knowledgeSourceName=ks-docs, kind=searchIndex, includeReferences=true; knowledgeSourceName=ks-web, kind=web, count=10, freshness=week"
  }
}
```

Response mirrors the REST contract (`response`, `references`, `activity`, `_status_code`). See [Knowledge Retrieval - Retrieve](https://learn.microsoft.com/en-us/rest/api/searchservice/knowledge-retrieval/retrieve?view=rest-searchservice-2025-11-01-preview) for schema details.

## More Details

See [MCP Server for Azure AI Search](https://heyjiqing.notion.site/MCP-Server-for-Azure-AI-Search-294de7b6e4e8805faccad1f60cc255e2?pvs=74)
