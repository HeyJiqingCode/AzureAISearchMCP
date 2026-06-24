---
name: azure-ai-search-mcp
description: Use this skill when a user wants help using the Azure AI Search MCP Server tools. Guide the user through available capabilities, required parameters, optional advanced settings, and Agentic Retrieval evidence configuration.
---

# Azure AI Search MCP

Use this skill as an interactive guide for the Azure AI Search MCP Server. Do not silently choose a tool when the user has not specified one. First explain the available capabilities, then ask the user to choose or describe their goal.

## Opening Prompt

When this skill is triggered and the user has not already chosen a tool, say something like:

```text
This MCP server exposes six Azure AI Search capabilities:

1. Keyword search - BM25 keyword retrieval over an index.
2. Semantic search - semantic ranking, captions, and optional extractive answers.
3. Vector search - vector-only similarity search using integrated vectorization.
4. Hybrid search - keyword plus vector retrieval.
5. Semantic hybrid search - hybrid retrieval plus semantic reranking.
6. Agentic Retrieval - Azure AI Search Knowledge Base retrieval with grounded answers.

Which capability do you want to use? You can also describe your goal, and I can suggest the closest tool.
```

If the user already clearly selected a tool or capability, skip the menu and ask only for missing required parameters.

## Capabilities And Required Parameters

### Keyword Search

Tool: `simple_search`

Required:

- `index_name`: Azure AI Search index name.
- `query`: keyword or simple query string.

Useful optional parameters:

- `top`: maximum results to return.
- `select`: comma-separated fields to return.
- `filter`: OData filter expression.
- `search_fields`: comma-separated searchable fields.

### Semantic Search

Tool: `semantic_search`

Required:

- `index_name`: Azure AI Search index name.
- `query`: natural-language query.
- `semantic_configuration`: semantic configuration name on the index.

Useful optional parameters:

- `top`: maximum results to return.
- `select`: comma-separated fields to return.
- `filter`: OData filter expression.
- `query_caption`: usually `"extractive"`.
- `query_answer`: set to `"extractive"` when the user wants semantic answers.

### Vector Search

Tool: `vector_search`

Required:

- `index_name`: Azure AI Search index name.
- `vector_fields`: comma-separated vector field names.
- `vector_text`: query text to vectorize using the index vectorizer.

Useful optional parameters:

- `k`: nearest neighbor count.
- `select`: comma-separated fields to return.
- `filter`: OData filter expression.

### Hybrid Search

Tool: `hybrid_search`

Required:

- `index_name`: Azure AI Search index name.
- `query`: keyword query.
- `vector_fields`: comma-separated vector field names.
- `vector_text`: query text to vectorize.

Useful optional parameters:

- `k`: vector candidate count.
- `top`: final result count.
- `select`: comma-separated fields to return.
- `filter`: OData filter expression.
- `search_fields`: comma-separated fields for keyword search.

### Semantic Hybrid Search

Tool: `semantic_hybrid_search`

Required:

- `index_name`: Azure AI Search index name.
- `query`: keyword or natural-language query.
- `vector_fields`: comma-separated vector field names.
- `semantic_configuration`: semantic configuration name on the index.
- `vector_text`: query text to vectorize.

Useful optional parameters:

- `k`: vector candidate count.
- `top`: final result count.
- `select`: comma-separated fields to return.
- `filter`: OData filter expression.
- `query_caption`: usually `"extractive"`.
- `query_answer`: set to `"extractive"` when semantic answers are useful.

### Agentic Retrieval

Tool: `agentic_retrieval`

Required:

- `knowledge_base_name`: Azure AI Search Knowledge Base name.
- `query`: user question.

Useful optional parameters:

- `reasoning_effort`: `"minimal"`, `"low"`, or `"medium"`. Default is `"low"`.
- `output_mode`: `"answerSynthesis"` or `"extractedData"`. Default is `"answerSynthesis"`.
- `max_runtime_seconds`: service-side runtime cap.
- `max_output_documents`: cap grounding document count.
- `knowledge_source_configs`: JSON object or JSON array encoded as a string.
- `include_diagnostics`: set to `true` only for troubleshooting.

Important constraint:

- Do not use `reasoning_effort="minimal"` with `output_mode="answerSynthesis"`. Use `low` or `medium` for answer synthesis, or use `minimal` with `extractedData`.

## Asking For Missing Parameters

Ask only for the missing values needed by the selected tool. Keep the question short.

Examples:

```text
To run semantic search, I need:
- index_name
- query
- semantic_configuration

What are those values?
```

```text
To run Agentic Retrieval, I need:
- knowledge_base_name
- query

Do you want source chunks returned in references[].content? If yes, I will help configure knowledge_source_configs with includeReferenceSourceData=true.
```

If the user gives enough information, proceed without asking them to choose again.

## Advanced Options

Only introduce `advanced_options` when the user asks for tuning, filtering behavior, thresholds, semantic timeout behavior, or hybrid/vector knobs.

`advanced_options` must be a JSON object encoded as a string. Do not use plain `key=value` text.

Allowed keys:

- Common: `debug`
- Semantic: `semantic_query`, `query_answer_count`, `query_answer_threshold`, `semantic_error_mode`, `semantic_max_wait_in_milliseconds`
- Vector: `exhaustive`, `weight`, `oversampling`, `filter_override`, `vector_filter_mode`, `vector_similarity_threshold`, `search_score_threshold`
- Hybrid: `max_text_recall_size`, `count_and_facet_mode`

Examples:

```json
{
  "advanced_options": "{\"vector_filter_mode\":\"preFilter\"}"
}
```

```json
{
  "advanced_options": "{\"max_text_recall_size\":100,\"vector_filter_mode\":\"preFilter\"}"
}
```

```json
{
  "advanced_options": "{\"semantic_error_mode\":\"partial\",\"semantic_max_wait_in_milliseconds\":3000}"
}
```

Use only one of `vector_similarity_threshold` or `search_score_threshold` in the same request.

## Agentic Source Data

For Agentic Retrieval, explain that `references[].content` is available only when Azure AI Search returns `sourceData`.

To request source chunks from a search index knowledge source, use:

```json
{
  "knowledge_source_configs": "{\"knowledgeSourceName\":\"ks-index\",\"kind\":\"searchIndex\",\"includeReferences\":true,\"includeReferenceSourceData\":true,\"maxOutputDocuments\":5}"
}
```

For multiple sources:

```json
{
  "knowledge_source_configs": "[{\"knowledgeSourceName\":\"ks-index\",\"kind\":\"searchIndex\",\"includeReferences\":true,\"includeReferenceSourceData\":true,\"maxOutputDocuments\":5},{\"knowledgeSourceName\":\"ks-web\",\"kind\":\"web\",\"includeReferences\":true,\"count\":5,\"freshness\":\"week\"}]"
}
```

## Reading Agentic Results

Default Agentic Retrieval response:

```json
{
  "answer": {
    "text": "... [ref_id:8] ...",
    "used_ref_ids": ["8"]
  },
  "references": [
    {
      "ref_id": "8",
      "source_type": "searchIndex",
      "title": "Reset VPN Password",
      "content": "Grounding chunk text when sourceData is available.",
      "document_id": "doc-1",
      "chunk_id": "chunk-7",
      "reranker_score": 3.13
    }
  ],
  "metadata": {
    "referenced_count": 1,
    "total_reference_count": 12
  }
}
```

Explain results like this:

- `answer.text` is the grounded answer.
- `answer.used_ref_ids` lists the cited reference IDs.
- Match `[ref_id:x]` in the answer to `references[].ref_id`.
- For `searchIndex`, prefer `content`, `chunk_id`, `document_id`, and `reranker_score`.
- For `web`, use `title` and `url`; `content` appears only if Azure returned source data.
- `metadata.referenced_count` is the number of cited references.
- `metadata.total_reference_count` is the total candidate references returned by Azure.

Use `include_diagnostics=true` only for debugging. It adds the SDK request, timeout budget, raw response, raw references, and activity details.

## Troubleshooting

| Symptom | Suggested response |
| --- | --- |
| User does not know which tool to use | Briefly explain the six capabilities and ask them to choose or describe their goal. |
| Missing required parameters | Ask only for the missing parameters for the selected tool. |
| `advanced_options must be a JSON object` | Tell the user `advanced_options` must be a JSON object encoded as a string. |
| Semantic query fails | Ask the user to verify the semantic configuration name, or suggest keyword/hybrid search. |
| Vector or hybrid query fails | Ask the user to verify `vector_fields` and integrated vectorization. |
| `references[].content` is missing | Explain that Azure did not return source data; suggest `includeReferenceSourceData=true`. |
| Agentic Retrieval times out | Suggest lowering `max_runtime_seconds` for service-side bounds or increasing server timeout environment variables. |

## Guardrails

- Do not expose keys, tokens, or `query_source_authorization` values in final answers.
- Do not enable `include_diagnostics` unless the user is debugging.
- Do not invent missing index names, vector fields, semantic configurations, or knowledge source names.
- Do not invent chunk text when `references[].content` is absent.
