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


```sh
curl -X POST "http://orca-infer.ai/api/v1/rag-go/generate-doc" \
     -H "Content-Type: application/json" \
     -d '{ "query_text": "How to run selenium test with specific version of chromedriver", "repo_id": "github.com/Blazemeter/taurus", "type":"direct", "limit": 15}'
```



Looking at the ingestion payload and the current retrieval path (`Query`/`QueryStandard` in `client_qdrant.go`, fan-out + budgeting in `rag_pipeline.go`), there's real room to improve retrieval quality. Here's what stands out, purely as analysis:

## 1. Retrieval is dense-only, no score filtering - DONE
`query()` builds a `QueryPoints` request with no `ScoreThreshold` and never inspects `hit.Score` beyond a debug log. That means:
- Every query always returns up to `limit` chunks even if the best match is a poor cosine match — irrelevant chunks get shipped to the LLM and consume budget/attention.
- **Improvement:** add a `ScoreThreshold` (tunable per collection) so low-similarity noise is dropped before it ever competes for the char budget in `AllocateChunkCharBudget`.

## 2. No hybrid (sparse + dense) retrieval - LAST
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

## 6. `date`/`month`/`date_short` are underused
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

---
**Priority order if you want the best effort/impact ratio:** (3) reranking → (1) score threshold → (4) neighbor-chunk stitching → (2) hybrid sparse+dense → (5) MMR/dedup → (8) query rewriting → (6)/(7)/(9)/(10) as refinements.
