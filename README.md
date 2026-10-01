## rag-trans_qdrant

**HOW THIS ALL WORKS**
```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             1. REPOSITORIES & GITOPS LIFECYCLE                                          │
│                                                                                                                         │
│  [ Developer Git Push ] ───> [ ImMnan/helm-rag_vLLM ] ───> [ ArgoCD Controller ] ───> Declares State & Reconciles Drift │
└─────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             2. KUBERNETES TARGET CLUSTER (HA NAMESPACE)                                 │
│                                                                                                                         │
│    [ External Ingress (HTTPS) ]                                                                                         │
│                 │                                                                                                       │
│                 ▼  (K8s Core Service Routing / ClusterIP)                                                               │
│        ┌──────────────────┐                                                                                             │
│        │  Hawk UI Pod     │                                                                                             │
│        │  (NextJS/Web)    │                                                                                             │
│        └────────┬─────────┘                                                                                             │
│                 │                                                                                                       │
│                 ▼ (Internal HTTP/REST Protocol Route)                                                                   │
│        ┌────────────────────────────────────────────────────────────────────────────────────────┐                       │
│        │  Hawk API Server (Routing & Auth Gateway Pod)                                          │                       │
│        └────────┬───────────────────────────────────────────────────────────────────────┬───────┘                       │
│                 │                                                                       │                               │
│                 │ (Low-latency Internal Pod-to-Pod Network)                             │ (gRPC Data Stream)            │
│                 ▼                                                                       ▼                               │
│  ┌──────────────────────────────────────────────────────┐                ┌───────────────────────────────┐              │
│  │ Orca Core Engine Pods (Written in Go)                │                │ Decoupled Embedding Model     │              │
│  │ [rag-trans_qdrant Workload Layer]                    │                │ (HuggingFace/Local Worker)    │              │
│  └──────────────┬────────────────────────┬──────────────┘                └──────────────┬────────────────┘              │
│                 │                        │                                              │                               │
│                 │ (Vector Search)        │ (Context Injected Prompt)                    │ (Generate Dense Vectors)      │
│                 ▼                        ▼                                              │                               │
│        ┌────────────────┐       ┌────────────────────────┐                              │                               │
│        │ Qdrant Pods    │       │ vLLM Inference Engine  │ <────────────────────────────┘                               │
│        │ (Vector DB)    │       │ (Distributed LLM Pods) │                                                              │
│        └────────┬───────┘       └───────────┬────────────┘                                                              │
│                 │                           │                                                                           │
└─────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────────────────┘
                  │                           │
                  ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                             3. PERSISTENT STORAGE LAYER                                                 │
│                                                                                                                         │
│   [ Dynamic CSI Provisioner ] ───> [ PersistentVolumeClaims (PVC) ] ───> Block Storage (Indexes / Weights Data Paths)   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

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

Standard change-summary requests (`type: "standard"`) retrieve all change chunks in
the requested date range without dropping chunks to fit the model context window. If
the change set is too large for one request, Orca splits it into batches, summarizes
each batch, and merges the partial summaries into one answer. 
Batch summaries and merge calls run concurrently, with at most four vLLM requests in flight per query.
If a batch still exceeds the context limit, Orca splits and retries that batch.
The response's `sources.change_batches` value reports the number of initial batches
used; it is `1` when all change context fits in one request.



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


## 1. Retrieval is dense-only, no score filtering - DONE
`query()` builds a `QueryPoints` request with no `ScoreThreshold` and never inspects `hit.Score` beyond a debug log. That means:
- Every query always returns up to `limit` chunks even if the best match is a poor cosine match — irrelevant chunks get shipped to the LLM and consume budget/attention.
- **Improvement:** add a `ScoreThreshold` (tunable per collection) so low-similarity noise is dropped before it ever competes for the char budget in `AllocateChunkCharBudget`.

## 2. No hybrid (sparse + dense) retrieval - DONE
The payload has no sparse/BM25-friendly field (e.g., no keyword/sparse vector), so retrieval is 100% dense embedding similarity. Dense embeddings are weak on exact identifiers — function names, error codes, env var names, config keys (`VLLM_TIMEOUT`, `EMBED_TIMEOUT`, specific commit SHAs) — which this codebase clearly cares about (per your own memory notes about exact-match filtering like `repo_id`).
- **Improvement:** add a sparse vector (Qdrant supports named sparse vectors) generated from BM25/SPLADE over `text`, and do a hybrid query (RRF fusion) in Qdrant. This alone often gives the largest retrieval-quality jump for code/config-heavy corpora.

## 3. No cross-encoder reranking - DONE 
Right now the top-K from Qdrant is used as-is. Bi-encoder similarity is good for recall, poor for precision at the top.
- **Improvement:** retrieve a larger candidate pool (e.g., `limit*4`) then rerank with a cross-encoder (or even use the LLM itself for cheap listwise reranking) before truncating to the final chunks that go into the prompt. This is usually the single highest-ROI change for RAG accuracy.

## 4. Chunking has no overlap/neighbor stitching, despite `chunk_index` existing - DONE
You store `chunk_index` per file but retrieval treats each chunk as an independent unit — no expansion to neighboring chunks. This causes truncated logic/functions ("mid-explanation" chunks) to be handed to the LLM without their context.
- **Improvement:** when a chunk from `file_path`+`chunk_index` scores well, fetch `chunk_index-1`/`chunk_index+1` from the same file (a Qdrant scroll/filter by `file_path` + range) and stitch them — classic "sentence-window"/"parent-document" retrieval pattern. Since `file_path` and `chunk_index` are already indexed in the payload, this is cheap to add.

## 5. No dedup/diversity (MMR) across results - DONE
If a file was chunked densely, multiple near-duplicate chunks from the same `file_path` can dominate all `limit` slots, crowding out other relevant files/components.
- **Improvement:** apply Maximal Marginal Relevance (MMR) or simple "cap N chunks per `file_path`" diversity logic post-retrieval.

## 6. `date`/`month`/`date_short` are underused - DONE
These exist for `QueryStandard`'s date-range filter, but nothing does recency-boosting for the default (non-"standard") path. For a repo where change history and code evolve, an old vs. new answer can matter.
- **Improvement:** for non-standard queries, consider a mild recency boost (e.g., combine similarity score with a decayed function of `date`) rather than a hard filter, so more recent commits/docs are favored without being mandatory.

## 7. `component` is a coarse-only filter, no weighting
`component` is used as an exact-match `must` filter, all-or-nothing. If it's wrong/missing at query time, retrieval silently returns 0 relevant results from that component, or if omitted, mixes everything.
- **Improvement:** consider `should` (boost) instead of `must` when the caller is uncertain, or fall back to unfiltered retrieval when a `must` component filter returns too few hits.

## 8. No query expansion / rewriting 
A single embed of the raw user query is used for both collections. Char-for-char the same vector is reused for change/code/doc collections which have very different content styles (diffs vs. source vs. prose).
- **Improvement:** generate collection-specific query variants (e.g., an LLM-rewritten "code-search style" query for the code collection vs. the raw NL query for docs) — this is a well-known technique (HyDE / query rewriting) that improves cross-domain recall.

## 9. Embedding model recorded but not used for staleness detection
`embedding_model` is stored per point but nothing checks it against the model actually configured for the live embedder. If you ever swap embedding models (e5 ↔ gte, see `client_factory.go`), old vectors and new query vectors become incomparable, silently degrading retrieval with no error.
- **Improvement:** validate/log a warning when the payload's `embedding_model` for retrieved hits doesn't match the active embedder, so a silent-quality-regression from model drift is caught rather than looking like "bad retrieval."

## 10. Payload indexing not verifiable from this code
Filtering on `repo_id`/`component`/date fields is only fast if those payload fields have Qdrant payload indexes created at collection setup — that's not part of this Go code, so it's worth double-checking the collection schema explicitly declares indexes for `repo_id`, `component`, and `date` (keyword/datetime index types) to avoid full scans as the collection grows.
