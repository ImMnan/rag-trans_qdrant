## rag-trans_qdrant

### Download the model from HuggingFace Hub & Uploading to GCP bucket

```bash
hf download Alibaba-NLP/gte-Qwen2-1.5B-instruct --local-dir /Desktop/gte/

gcloud storage cp -r /Desktop/gte/* gs://rag-model-weights-hawk/gte-model/
```

### Build and publish Orca

The `VERSION` argument is required. It is used as both the Docker image tag and
the Orca version reported in the application startup log.

Build the container image locally:

```bash
make build-image VERSION=1.2.3
```

Push an image that has already been built to Docker Hub:

```bash
docker login
make push-image VERSION=1.2.3
```

Build and push the image in one command:

```bash
docker login
make release VERSION=1.2.3
```

These commands publish the image as `immnan/orca-rag:1.2.3`. Replace `1.2.3`
with the tag you want to release.

To build only the Linux Go binary with the same embedded version:

```bash
make build-binary VERSION=1.2.3
```

List all available targets or remove local binary artifacts:

```bash
make help
make clean
```

### Retrieval diversity

Qdrant retrieval uses maximal marginal relevance (MMR) by default. It fetches
`QDRANT_MMR_OVERFETCH` candidates, removes exact normalized-text duplicates,
then selects the requested number of chunks using the relevance/diversity
balance in `QDRANT_MMR_LAMBDA` (`1.0` is relevance-only, `0.0` is
diversity-only). Set `QDRANT_MMR_ENABLED=false` to preserve the original ranked
results.

Context-budget truncation is enabled by default. Set
`CONTEXT_TRUNCATION_ENABLED=false` to pass all retrieved chunks to the prompt;
this can exceed the model context window. Each response reports
`meta.context_truncation_enabled` and `meta.context_truncation_active`, where
the latter is true only when the request actually dropped chunks.


```sh
curl -X POST "http://orca-infer.ai/api/v1/rag-go/generate-doc" \
     -H "Content-Type: application/json" \
     -d '{ "query_text": "How to run selenium test with specific version of chromedriver", "repo_id": "github.com/Blazemeter/taurus", "type":"direct", "limit": 15}'
```

### Progress streaming

Both RAG endpoints support Server-Sent Events (SSE) when `stream=true` is present:

```sh
curl -N -X POST "http://orca-infer.ai/api/v1/rag-go?stream=true" \
     -H "Content-Type: application/json" \
     -d '{"query_text":"What changed?","repo_id":"github.com/example/repo","type":"direct"}'
```

The response emits `progress` events followed by exactly one terminal `complete` or
`error` event. A `progress` event uses this stable shape:

```json
{"stage":"qdrant_query_complete","message":"Qdrant query complete","percent":35}
```

The current milestones are `request_received` (0), `embedding_complete` (15),
`qdrant_query_complete` (35), `messages_assembled` (60; document composition uses
80), and `vllm_complete` (95). `complete` contains the same response JSON returned
by a normal request.

To change a percentage or add a progress item, call the request-scoped reporter at
the completed pipeline boundary:

```go
req.ReportProgress("rerank_complete", "Reranking complete", 50)
```

Use a stable, snake-case stage name and a percentage that never decreases within a
single workflow. The handler owns the terminal `complete` event, so pipeline code
should only report work milestones. Without `stream=true`, the endpoints retain the
original single JSON response.



### Hybrid retrieval

The code, change, and documentation collections use named Qdrant vectors: `dense`
for the existing cosine embedding and `sparse` for BM25. The ingestion pipeline
generates sparse vectors with `fastembed`'s `Qdrant/bm25` model. The GTE embedding
service exposes the matching query encoder at `/sparse`; e5 currently uses
dense-only retrieval.

For code and document retrieval, Orca sends dense and sparse candidate searches to
Qdrant and combines them with reciprocal rank fusion (RRF). This can improve recall
for exact identifiers such as function names, error messages, environment variables,
and configuration keys while retaining dense semantic matching. Both candidate
searches apply the same `repo_id` and optional `component` filters.

`QDRANT_SCORE_THRESHOLD` is a cosine-similarity threshold, so hybrid queries apply
it only to the dense candidate search. Sparse BM25 scores and fused RRF scores are
not cosine similarities and are not filtered by this threshold. If sparse query
encoding fails, that request logs a warning and falls back to dense-only retrieval.

When MMR is enabled, hybrid-result relevance follows the fused result rank; cosine
similarity between dense vectors is still used to diversify the selected chunks.
This preserves sparse-only matches that could otherwise be pushed down by dense
similarity. Neighbor-chunk stitching, optional cross-encoder reranking, and context
truncation remain downstream of retrieval. Cross-encoder reranking is controlled by
`RERANK_ENABLED` and is disabled by default.

Hybrid retrieval is enabled by default with `QDRANT_HYBRID_ENABLED=true`. The
related Qdrant settings are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `QDRANT_HYBRID_ENABLED` | `true` | Request BM25 query vectors and use hybrid retrieval when supported by the embed client. |
| `QDRANT_DENSE_VECTOR_NAME` | `dense` | Named dense vector; set to an empty value only for legacy collections with an unnamed dense vector. |
| `QDRANT_SPARSE_VECTOR_NAME` | `sparse` | Named sparse vector in Qdrant. |
| `QDRANT_HYBRID_PREFETCH_MULTIPLIER` | `2` | Multiplier for each dense/sparse candidate pool before RRF fusion. |

Hybrid retrieval requires collections created with both named vectors and points
ingested with sparse vectors. Recreating an empty collection does not backfill
existing points: rerun ingestion for the code, change, and documentation collections
using the ingestion repository's full-reingest procedure before enabling hybrid
queries. Standard change-summary requests continue to scroll all change chunks in
the requested date range rather than using vector search.
