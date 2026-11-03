# Orca: A Source-Guided Learning Guide

This guide explains the current `rag-go` implementation from startup to the final answer. It is intended for learning, debugging, and planning changes. Descriptions refer to the checked-in source, not to an assumed production deployment. Configuration and external model behavior can differ in your environment.

### Reading Routes

- First pass: [mental model](#1-the-mental-model), [startup](#2-entry-point-and-service-lifetime), [HTTP API](#3-http-boundary-routes-validation-and-dispatch), then [worked examples](#14-worked-requests-and-response-interpretation).
- Retrieval deep dive: [interfaces](#4-interfaces-and-structs-how-the-parts-fit-together), [embedding](#5-creating-query-vectors-go-to-python), and [Qdrant](#6-qdrant-from-a-query-vector-to-context-text).
- Follow one branch: [current code](#7-path-a-answering-from-current-code), [standard changes](#8-path-b-standard-change-summaries), or [documents](#9-path-c-generate-or-update-documentation).
- Generation behavior: [prompts](#10-prompt-construction-where-answer-behavior-is-controlled), [budgets](#11-context-and-output-token-budgets), and [vLLM](#12-vllm-generating-the-final-text).
- Making changes: [configuration](#13-configuration-and-deployment-reference), [change map](#15-where-to-make-a-change), [efficiency](#16-efficiency-tuning-without-guessing), [debugging](#17-debugging-and-verification), and [limitations](#18-current-boundaries-and-caveats).

## 1. The Mental Model

Orca is a retrieval and answer orchestration service. It accepts a question, obtains relevant stored material from Qdrant, constructs chat messages, and asks a vLLM-hosted model to write an answer.

```text
Caller
	|
	v
Fiber HTTP API (Go)
	|
	+-- ordinary answer --> embed question --> search code_chunks --+
	|                                                             |
	+-- standard answer --> scroll matching change_chunks ---------+--> prompt --> vLLM --> answer
	|                                                             |
	+-- generate document --> retrieve code + existing documents ---+
```

Three distinctions prevent a lot of confusion:

1. **Ingestion is not query-time retrieval.** Orca's Go service reads existing Qdrant collections. It does not clone a repository, split its files into chunks, embed every source file, or populate those collections.
2. **A vector is not the answer context.** A query vector helps locate relevant records. The stored text and metadata returned from those records become the model's context.
3. **Not every path embeds a question.** The standard change-summary path deliberately reads all matching change records rather than performing semantic top-k retrieval.

Useful vocabulary:

| Term | Plain-language meaning |
| --- | --- |
| Chunk | A stored piece of source code, change history, or documentation. |
| Dense vector | A list of floating-point numbers representing semantic meaning. |
| Sparse vector | Selected numeric feature IDs and their weights, useful for lexical matching. |
| Payload | Qdrant record metadata and text, separate from its vector. |
| Retrieval | Selecting stored chunks to give the model. |
| Reranking | Scoring retrieved candidates again against the actual question. |
| Prompt | The system and user chat messages sent to the language model. |
| Completion | The model's generated answer. |

## 2. Entry Point and Service Lifetime

Start reading at [rag-go/main/main.go](rag-go/main/main.go). Its `main()` function is the composition root: it constructs concrete clients and injects them into pipelines.

```text
main()
	|
	+-- configure zerolog
	+-- loadConfig()                         environment -> config struct
	+-- qdrant.NewClient(...)                Qdrant gRPC client
	+-- embedder.NewClientFromType(...)      embedding/reranking HTTP client
	+-- optional SparseEmbedder assertion    enable sparse query encoding
	+-- vllm.NewHTTPClient(...)              OpenAI-compatible HTTP client
	+-- pipeline.New(...)                   RAGPipeline
	+-- pipeline.NewDoc(...)                DOCPipeline
	+-- fiber.New(...)                      HTTP server with timeouts
	+-- handler.Register(...)               register the routes
	+-- install SIGINT/SIGTERM handler
	+-- app.Listen(...)                     serve requests
```

`LOG_VERBOSE=true` selects debug logging; otherwise the log level is info. Logs are JSON with timestamps and `service: "orca"`. `orcaVersion` defaults to `dev` and can be replaced at build time.

The same embedding client is passed as both `Embedder` and `Reranker`. These are different capabilities exposed by one service, not the same operation. Hybrid retrieval is enabled only when configured and the client implements `SparseEmbedder`.

On SIGINT or SIGTERM, the server attempts graceful shutdown with a 30-second context. Client construction is not an end-to-end dependency readiness check.

## 3. HTTP Boundary: Routes, Validation, and Dispatch

The route owner is [rag-go/pkg/handler/ragHandler.go](rag-go/pkg/handler/ragHandler.go).

| Method and route | Handler | Purpose |
| --- | --- | --- |
| `GET /health` | `handleHealth` | Returns `{"status":"ok","service":"orca"}`. Does not probe downstream services. |
| `POST /api/v1/rag-go` | `ragHandler.handleRAG` | Answers about current code or summarizes changes. |
| `POST /api/v1/rag-go/generate-doc` | `docHandler.handleRAG` | Generates or updates document text from retrieved evidence. |

The answer endpoint binds JSON into `RAGRequest`, validates it, copies supported fields into `pipeline.Request`, and calls `RAGPipeline.Execute`.

```text
HTTP JSON -> RAGRequest -> validation/defaults -> pipeline.Request
																										 |
																										 v
																							RAGPipeline.Execute
																										 |
																										 v
HTTP JSON <- pipeline.Response <----------------------+
```

### 3.1 Request Fields

| Field | Meaning and actual handling |
| --- | --- |
| `query_text` | Required; the handler checks for an empty string, not whitespace-only text. |
| `repo_id` | Repository filter. At least this or `component` must be supplied. |
| `repo_name` | Accepted in the HTTP structs but not copied into the pipeline request or used for filtering. It cannot replace `repo_id`. |
| `component` | Optional scope; component-only requests are accepted. |
| `type` | Case-insensitive, trimmed `standard` selects change summaries; other values, including empty, take the current-code path. |
| `limit` | Defaults to 5 for answers, 25 for document generation when zero or negative. Not a limit on the standard path's complete change scan. |
| `token_limit` | Input to output-token budget selection, not a retrieved-chunk count. |
| `from_date`, `to_date` | Answer endpoint only. Must be valid `YYYY-MM-DD`; from must not exceed to. Used for standard change retrieval, not current-code retrieval. |

When both dates are absent, the answer handler fills in today and 30 calendar days earlier. A single supplied bound is left one-sided. The standard path receives these bounds; the current-code path does not use them.

Invalid JSON, missing required scope/question, invalid dates, and invalid `stream` values return HTTP 400 before execution. Pipeline errors return HTTP 500 with a deliberately generic error message, except for the recognized oversized Qdrant gRPC response error.

### 3.2 Streaming Means Progress, Not Token Streaming

Append `?stream=true` to either POST route to receive Server-Sent Events (SSE). Empty or `false` selects normal JSON; other query parameter values are rejected. The request body does not select streaming.

```text
event: progress
data: {"stage":"request_received","message":"Request received","percent":0}

... completed pipeline milestones ...

event: complete
data: {"answer":"...","type":"...","sources":{...},"meta":{...}}
```

The handler flushes each event, disables caching, and sets `X-Accel-Buffering: no`. A pipeline failure after streaming begins produces an `error` event, not a replacement HTTP 500 response. The model's answer still arrives as one completed string; this is not vLLM token-by-token streaming. Write and flush errors are ignored by `writeSSE`, so do not assume this provides robust disconnect handling.

## 4. Interfaces and Structs: How the Parts Fit Together

The contracts and orchestration structs live in [rag-go/pkg/pipeline/ragPipeline.go](rag-go/pkg/pipeline/ragPipeline.go). In Go, a concrete type implements an interface by having the required methods; there is no explicit `implements` declaration.

```text
										 pipeline owns these contracts
										 =============================
QdrantQuerier  <----- qdrant.Client
VLLMCompleter <----- vllm.HTTPClient
Embedder       <----- embedder.Client OR embedder.GTEQwenClient
Reranker       <----- embedder.Client OR embedder.GTEQwenClient
SparseEmbedder <----- embedder.GTEQwenClient only

The pipelines know the interfaces, not how HTTP or gRPC works.
Tests can substitute small structs with matching methods.
```

| Interface | Methods and purpose |
| --- | --- |
| `QdrantQuerier` | `Query`: code/doc vector retrieval. `QueryStandardAll`: exhaustive filtered change scan. `QueryStandard`: vector retrieval with a date condition, available but not used by the active standard execution path. |
| `Embedder` | `Embed(ctx, text) -> ([]float32, error)`: dense question encoding. |
| `SparseEmbedder` | `SparseEmbed(ctx, text) -> ([]uint32, []float32, error)`: sparse feature IDs and weights. |
| `Reranker` | `Rerank(ctx, query, documents) -> ([]float32, error)`: one relevance score per candidate, in input order. |
| `VLLMCompleter` | `Complete(ctx, messages, maxTokens) -> (string, error)`: one complete generated response. |
| `DocProcessor` | `Process(ctx, req, codeChunks, docChunks) -> (string, error)`: document workflow, declared in [rag-go/pkg/pipeline/docWorkflow.go](rag-go/pkg/pipeline/docWorkflow.go). |
| `DocDecisionEngine` | `Decide` and `ComputeConfidence`: local policy after the model's triage and composition. |

`ctx` is Go's `context.Context`: downstream requests can observe cancellation or deadlines through it. The HTTP handler passes the Fiber request context. This code does not install a separate overall pipeline deadline; client timeouts and server timeouts are distinct controls.

### 4.1 Important Data Shapes

```text
Request                         internal, not itself the HTTP JSON schema
	QueryText                     user question
	RepoID, Component             retrieval scope
	Type                          standard vs all other answer requests
	Limit, TokenLimit             chunk selection vs generated-token budget
	FromDate, ToDate               standard reporting window
	AppProfile                    filled by pipeline where applicable
	RepoName                      declared, but not wired into HTTP dispatch
	Progress                      optional callback

Message
	Role                          "system" / "user"
	Content                       plain text including retrieved evidence

Response                        outer JSON object
	Answer                        string (document workflow embeds JSON here)
	Type                          incoming type, or empty in document responses
	Sources                       map of counts, not a list of source records
	Meta
		RepoID, Component, QueryText
		ContextTruncationEnabled
		ContextTruncationActive
```

`RAGPipeline` holds the Qdrant, vLLM, embedding, sparse, and reranking dependencies plus collection names, profile settings, context settings, and a logger. `DOCPipeline` has similar dependencies, but uses code/doc collections and adds a `DocProcessor`; it does not retrieve change history.

`New` and `NewDoc` construct these values. `WithLogger` and `WithSparseEmbedder` set optional dependencies and return the same pipeline pointer for chaining. `NewDoc` installs `LLMDocProcessor` with `DefaultDocDecisionEngine` automatically.

The embedder factory's `EmbeddingClient` interface guarantees `Embed` and `Rerank`, not `SparseEmbed`. Startup uses a type assertion to discover the optional sparse capability of its concrete value.

## 5. Creating Query Vectors: Go to Python

### 5.1 What Is Encoded, and When?

Only the question is encoded by these Go query paths. Repository chunk vectors already need to exist in Qdrant from a separate ingestion process.

```text
"How do I configure the service?"
			 |
			 +-- Embed ----------------> dense: [0.014, -0.28, ...]
			 |
			 +-- SparseEmbed (optional) -> indices: [featureA, featureB, ...]
																		 values: [weightA, weightB, ...]
```

The values above are illustrative, not an actual model result. Dense dimensions depend on the selected model and must match the collection's vector configuration. Sparse feature IDs are not positions in a short dense list; each identifies a feature in the sparse model's vocabulary/encoding scheme.

The local `queryVectors` struct keeps `dense`, `sparseIndices`, and `sparseValues` together. `embedQuery` starts sparse encoding in a goroutine and runs dense encoding in the current goroutine, then waits for both.

```text
embedQuery
	+-- goroutine: sparse.SparseEmbed(question) --+
	+-- current:   embedder.Embed(question) ------+--> wait --> queryVectors

dense error  -> fail request
sparse error -> warning, clear sparse arrays, continue dense-only
```

Because it waits for the sparse goroutine before returning, even a failed dense request can wait for the sparse call to finish. These are concurrent network requests, not a combined dense+sparse batch endpoint.

### 5.2 Go Client Selection and HTTP Contracts

The factory is [rag-go/pkg/embedder/clientFactory.go](rag-go/pkg/embedder/clientFactory.go). `EMBED_SERVICE_TYPE` is required in `<type>:<host>` form, for example `gte-qwen:gte-embed-service:80`. Supported types are `e5` and `gte-qwen`, normalized for case and surrounding whitespace by the factory. An unknown type terminates startup through fatal logging.

| Capability | E5 Go client | GTE Go client |
| --- | --- | --- |
| Dense embedding | `POST /embed` with `{"text":"question"}` | `POST /embed` with `{"text":"question","input_type":"query"}` |
| Response | `{"vector":[...]}` | `{"vector":[...]}` |
| Rerank | `POST /rerank` with `query` and `documents` | Same |
| Rerank response | `{"scores":[...]}` | Same |
| Sparse query | Not implemented | `POST /sparse` with `{"text":"question"}` |
| Sparse response | Not applicable | `{"indices":[...],"values":[...]}` |

Implementation: [rag-go/pkg/embedder/clientE5-large.go](rag-go/pkg/embedder/clientE5-large.go) and [rag-go/pkg/embedder/clientGte-qwen.go](rag-go/pkg/embedder/clientGte-qwen.go).

Both clients use request contexts, JSON bodies, a configured HTTP timeout, and a reusable HTTP transport with up to 50 idle connections per host. They require HTTP 200. They do not implement general retry/backoff. The GTE sparse client checks that index and weight counts match. Dense responses are decoded but not explicitly checked for empty vectors or expected dimension in the Go client.

### 5.3 What the Python Services Actually Do

These are source-level descriptions; they do not assert that models are installed or running locally.

| Behavior | [embed-service/e5-main.py](embed-service/e5-main.py) | [embed-service/gte-main.py](embed-service/gte-main.py) |
| --- | --- | --- |
| Default dense model | `intfloat/multilingual-e5-large` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` |
| Query formatting | Prefixes input with `query: ` | Uses `EMBED_QUERY_TEMPLATE`, default `{text}` |
| Document formatting | No distinct document API in this file | Uses `EMBED_DOCUMENT_TEMPLATE`, default `{text}` |
| Normalization | `normalize_embeddings=True` | `normalize_embeddings=True` |
| Model lifetime | Loaded once during FastAPI lifespan startup | Same |
| Reranker | Optional `RERANKER_MODEL` cross-encoder | Same |
| Sparse model | None | `SPARSE_MODEL`, default `Qdrant/bm25` |
| Device selection | Delegated to library defaults | `EMBED_DEVICE=auto/cpu/cuda`; auto selects CUDA if available, otherwise CPU |

`resolve_model_path` looks for the cached Hugging Face `refs/main` snapshot first, then a sorted available snapshot directory, then returns the original model name. Dense/reranker loading uses `local_files_only=True`: provisioning model files is a deployment responsibility. GTE additionally requires a resolved local sparse model directory when sparse encoding is enabled; missing it raises a startup error. An explicitly empty `SPARSE_MODEL` disables sparse loading and leaves `/sparse` returning 503.

The async FastAPI endpoints move blocking model inference to `run_in_threadpool`. This avoids executing those model calls directly on the async event loop; it does not mean unlimited safe GPU parallelism or automatic batching.

The cross-encoder receives `[question, candidate_text]` pairs and returns scores. If `RERANKER_MODEL` was not loaded, `/rerank` returns 503; the Go pipeline's reranking helper falls back to retrieval order.

GTE also exposes `POST /v1/embeddings` for OpenAI-compatible document/batch embedding. It formats inputs as **documents**, returns `data[].embedding`, and is not called by Orca's Go request path. Its existence helps an external ingester use this service, but it does not itself insert anything into Qdrant. Blank entries are filtered before indexing the response, which matters to a batch-ingestion caller maintaining input/output alignment.

### 5.4 Compatibility Rule When Changing Models

```text
ingestion model + preprocessing + vector dimensions
											 must agree with
query model + preprocessing + collection vector schema
```

Using a different query model with an old collection can fail dimension checks or produce poor retrieval even when dimensions happen to match. Query and document templates may intentionally differ for an instruct model, but must form the intended matching pair. Sparse ingestion and query encoding must use compatible feature IDs/weights; the collection must also have its sparse vector configuration set appropriately. Switching `EMBED_SERVICE_TYPE` alone does not migrate stored vectors.

## 6. Qdrant: From a Query Vector to Context Text

The implementation is [rag-go/pkg/qdrant/clientQdrant.go](rag-go/pkg/qdrant/clientQdrant.go). This client uses Qdrant gRPC, normally port 6334, with insecure transport credentials. vLLM and embedding use HTTP instead. These transports are independent.

### 6.1 Expected Payload and Vector Layout

Conceptually, a stored code/doc point looks like this; it is an example of the fields the reader expects, not an ingestion schema enforced by Orca:

```json
{
	"id": "a-point-id",
	"vector": {
		"dense": [0.1, 0.2],
		"sparse": {"indices": [123, 456], "values": [0.8, 1.2]}
	},
	"payload": {
		"repo_id": "github.com/example/project",
		"component": "api",
		"file_path": "src/server.go",
		"chunk_index": 7,
		"chunk_text": "stored source text"
	}
}
```

The two dense numbers are only illustrative. Actual vector dimensions, IDs, and sparse schema come from ingestion/Qdrant setup.

| Field | How the reader uses it |
| --- | --- |
| `repo_id` | Exact-match repository filter. |
| `component` | Exact-match component filter when supplied. |
| `chunk_text`, then `text` | Code/doc text fallback order. Points with no nonempty text are skipped. |
| `change_chunk`, then `chunk_text`, then `text` | Standard scroll text fallback order. |
| `file_path`, then `doc_ref` | Source attribution, rendered as `[source: ...]`. |
| `chunk_index` | Integer position used for adjacent-chunk retrieval. A string value is not accepted as an integer. |
| `date`, configured date field, `date_short` | Used by the standard date-window scan as described below. |
| Dense vector | Used for semantic search and, when requested, local MMR diversity comparison. |
| Sparse vector | Used by the sparse search branch of hybrid retrieval. |

`filterConditions` combines supplied scope fields with AND semantics:

```text
repo_id supplied?   -> payload.repo_id == request.repo_id
component supplied? -> payload.component == request.component
both supplied?      -> both must match
component only?     -> can match that component across repositories
```

There is no `repo_name` lookup and no authorization check here. A retrieval filter is not a tenant authorization policy; enforce access separately when exposing the service.

### 6.2 Dense-Only vs Hybrid Search

```text
Query(...) -> query(...)
								|
								+-- no usable sparse arrays/name --> dense nearest-neighbor query
								|
								+-- usable sparse arrays + name
											|
											+-- dense prefetch --+
											+-- sparse prefetch -+--> Qdrant RRF fusion
																								|
																								v
																	parse payloads and returned vectors
																								|
																								v
																			optional local MMR
																								|
																								v
																	optional neighbor stitching
																								|
																								v
																	[]string with source prefixes
```

Dense search targets `QDRANT_DENSE_VECTOR_NAME`, default `dense`. Explicitly setting it to an empty string selects a legacy unnamed dense vector. Sparse search targets `QDRANT_SPARSE_VECTOR_NAME`, default `sparse`.

Hybrid is selected only when the sparse vector name is nonempty, indices are nonempty, and the index/value lengths match. Qdrant receives separate dense and sparse prefetches with the same filters, followed by reciprocal rank fusion (RRF). RRF merges ranked lists; it is not averaging raw dense and sparse scores. This lets exact identifiers found by sparse search survive alongside semantic matches.

`QDRANT_SCORE_THRESHOLD > 0` applies a threshold to dense search. In hybrid mode it applies only to the **dense prefetch branch**, not to the sparse branch or fused RRF scores. A dense-score threshold is therefore not a universal hybrid relevance threshold.

### 6.3 Candidate Multipliers: Why a Limit of 5 Can Do More Work

Let `L` be the requested limit, `R` the reranking multiplier when enabled, `M` the MMR multiplier when enabled, and `H` the hybrid prefetch multiplier:

```text
pipeline asks Qdrant for      L * R
Qdrant fused candidate cap   L * R * M
each hybrid prefetch cap     L * R * M * H
MMR returns at most          L * R
reranker retains at most     L
```

Disabled stages contribute a multiplier of 1. These are requested caps, not guaranteed hit counts or an exact measure of server computation.

For `L=5`, enabled reranking with `R=4`, MMR `M=3`, and hybrid `H=2`, the pipeline requests 20 candidates, Qdrant's fused cap is 60, and each prefetch can request 120. MMR reduces to at most 20; cross-encoder reranking keeps 5. With source defaults, reranking is **off**, so the corresponding caps are 5, 15, and 30.

### 6.4 MMR: Prefer Relevant Evidence Without Repeating It

MMR means maximal marginal relevance. `selectDiverse` balances query relevance against similarity to already-selected chunks:

```text
candidate value = lambda * relevance
									- (1 - lambda) * greatest_similarity_to_selected

repeat: choose the best remaining candidate until the limit is reached
```

Dense-mode relevance is cosine similarity to the query when a usable vector is returned; otherwise it falls back to Qdrant's score. Hybrid-mode relevance uses normalized fused ranking, so dense similarity does not immediately bury sparse-only exact matches. Dense cosine is still used for the diversity penalty.

`lambda=1` favors relevance only; lower values favor diversity more. The default is 0.7. The client requests returned dense vectors for MMR, increasing response size and local computation.

`deduplicateHits` normalizes case and whitespace in chunk text before dropping duplicates; it does not include the source path in the duplicate key. Identical text from different files may therefore collapse. With missing vectors, no useful vector-based diversity penalty is available. If candidate count is already at or below the limit, the helper deduplicates without a greedy selection pass. When MMR is disabled, this deduplication is not a separate always-on stage.

### 6.5 Neighbor Stitching: Restore Local Continuity

With `QDRANT_NEIGHBOR_STITCH=true`, selected hits with a source and integer `chunk_index` can be expanded using preceding/following chunks:

```text
hit: src/server.go, index 7

filter-only scroll for missing indices 6 and 8
							|
							v
one returned context string:
[source: src/server.go]
text of chunk 6

text of chunk 7

text of chunk 8
```

The implementation batches missing neighbor indices once per source file within one query result, rather than making a request per hit. It avoids stitching neighbors already selected as primary hits. The lookup uses exact `file_path`, repository/component scope, and integer indices; a `doc_ref` fallback used for attribution will not help unless that value is also present as `file_path` in the stored records.

This is best effort: scroll errors produce warnings and leave the affected hit unchanged. Stitching increases **text length**, not the count of returned strings, and happens before pipeline cross-encoder reranking. It does not guarantee perfect deduplication of overlapping expansions or whole-function boundaries. Large stitched text can increase reranking cost and consume context budget.

### 6.6 Standard Retrieval Is a Different Algorithm

`QueryStandardAll` does not call the vector-query path. It uses filter-only `Scroll` requests, each requesting 256 points, follows page offsets, and accumulates all matching text in memory.

```text
for each supported date field:
	scroll page -> read payloads -> deduplicate -> follow next offset
			 |
			 +-- until all pages are read

return every matching change chunk
```

Date-field attempts are the configured field, `date`, and `date_short`, deduplicated in that order. `month` is skipped to avoid coarse month-level filtering. Timestamp fields use a UTC range from the start of `from_date` through the last nanosecond of `to_date`. `date_short` uses a keyword match against every day string in the range. Results from these field scans are unioned and deduplicated by point ID, with a source/text fallback when no ID exists.

There is no semantic top-k selection, score threshold, MMR, neighbor stitching, reranking, or pipeline context truncation on this active standard path. This preserves the reporting window's input coverage, at the cost of potentially large retrievals and many model calls. Pagination order is not explicitly sorted by event date here.

**Date-bound caveat:** `standardDateFilters` parses both dates unconditionally. Although the HTTP handler accepts a single supplied bound, the absent bound then fails in retrieval. Supply both bounds, or omit both to obtain the handler's default window. This is current behavior, not a supported one-sided standard query.

`QueryStandard` remains in the interface and client, but `RAGPipeline.Execute` calls `QueryStandardAll`. Also, the comment above `QueryStandardAll` mentions comparing against vector-ranked code; the active standard branch does not retrieve code. Follow executable calls when comments disagree.

## 7. Path A: Answering from Current Code

Owner: `RAGPipeline.Execute` in [rag-go/pkg/pipeline/ragPipeline.go](rag-go/pkg/pipeline/ragPipeline.go).

```text
type != "standard" (after trimming/case normalization)
	|
	+-- resolveAppProfile(repo_id)
	+-- embedQuery(query_text)
	+-- choose queryLimit (overfetch for reranker if enabled)
	+-- qdrant.Query(codeCollection, vectors, scope, queryLimit)
	+-- optional rerankChunks(..., keep=limit)
	+-- optional TruncateChunksToCharBudget
	+-- annotateCodeChunks
	+-- buildPrompt
	+-- ResolveTokenBudget
	+-- vllm.Complete
	+-- assemble Response
```

Only the code collection is queried. The question does not search `doc_chunks` or `change_chunks` on this path. Date fields do not affect retrieval here.

### 7.1 Optional Cross-Encoder Reranking

[rag-go/pkg/pipeline/rerank.go](rag-go/pkg/pipeline/rerank.go) contains `rerankChunks`:

1. Send the query and retrieved candidate strings to the reranker.
2. Require one score per candidate.
3. Stable-sort descending by score and retain at most `limit` candidates.
4. On service error, wrong score count, or nil reranker, keep the head of the existing retrieval order instead.

Stable sorting preserves prior order for tied scores. Reranking is a query-document relevance stage, not the MMR diversity stage, and it operates on the text after neighbor stitching.

### 7.2 Evidence Labels

[rag-go/pkg/pipeline/evidence.go](rag-go/pkg/pipeline/evidence.go) adds a label to each retained code string:

```text
[evidence: mixed]
[source: src/server.go]
// Opens the configured listener.
func startServer(...) { ... }
```

`classifyCodeChunk` uses line/block-comment heuristics to choose `implementation`, `mixed`, or `comment-only`. It ignores source/evidence labels and recognizes several languages' comment prefixes. This is not an AST parser, reachability proof, or reliable classifier for every config/prose language. It cannot guarantee the prompt's assumption that chunks are whole functions.

The prompt tells the model to trust implementation over comments, cite supplied paths, preserve identifiers, and avoid assuming a defined function is enabled/reachable. These are grounding instructions, not formal verification. This annotation step is used by the current-code RAG answer path; the document pipeline does not call `annotateCodeChunks`.

### 7.3 Application Profiles

`resolveAppProfile` selects a filename using an exact `repo_id` key in `AppProfileFiles`, then [rag-go/pkg/pipeline/appProfile.go](rag-go/pkg/pipeline/appProfile.go) reads it under `APP_PROFILE_DIR`.

`filepath.Base(profileFile)` keeps the configured filename from introducing directory traversal. Missing/unconfigured profiles return an empty string; read failures are warned about and do not fail the request. The profile is loaded per request and supplies product/audience context, not retrieval filters or new source evidence. Component-only requests normally have no repository profile mapping.

## 8. Path B: Standard Change Summaries

This path is intended to summarize what changed within a reporting window, not to answer implementation questions from the current code snapshot.

```text
type == "standard"
	|
	+-- QueryStandardAll(changeCollection, scope, dates)
	+-- answerStandard(all change chunks)
				|
				+-- packChunksByTokens
				+-- parallel batch summaries (map)
				+-- merge summaries in rounds (reduce)
	+-- validate required headings
	+-- at most one format-repair call
	+-- Response with change count and initial batch count
```

The batching implementation is [rag-go/pkg/pipeline/standardParallel.go](rag-go/pkg/pipeline/standardParallel.go), using helpers from [rag-go/pkg/pipeline/tokenBudget.go](rag-go/pkg/pipeline/tokenBudget.go).

### 8.1 Map/Reduce for a Large Reporting Window

```text
all change text
		|
		v
[batch 0] [batch 1] [batch 2] ...
		|         |         |
		v         v         v
[summary0][summary1][summary2] ...    map: bounded concurrent vLLM calls
			 \      |      /
				\     |     /
					merge groups              reduce: repeat until one summary remains
								|
								v
					 final answer
```

`packChunksByTokens` keeps all input text rather than dropping a tail. An oversized chunk is split near line boundaries, with a UTF-8 boundary fallback for long lines. It always returns at least one batch, even for no retrieved changes, so the empty-window path still calls the model with a no-data prompt.

`summarizeStandardBatch` calls the normal standard prompt. If the error text contains `maximum context length` and the batch has at least two pieces, it halves the batch and recursively summarizes each half. A single still-oversized piece fails instead of retrying forever. Other errors fail immediately.

`mergeStandardPartials` groups partial summaries into fitting merge prompts and repeats until only one remains. If normal packing would produce one group per partial and make no progress, it forces pairwise groups. This guarantees fewer partials if calls succeed, but forced pairs can still overflow the actual model context; merge errors propagate without the batch-splitting recovery used by `summarizeStandardBatch`.

`runParallel` uses a semaphore and permits at most **10 in-flight vLLM calls per standard request**. The first error cancels its child context; it waits for started work and returns the first error. This is not a global semaphore: simultaneous incoming requests can collectively exceed 10 calls. Results are stored by batch index, preserving assembly order despite out-of-order completion. A mutex protects concurrent batch-progress reports because the SSE writer is not goroutine-safe.

### 8.2 Required Sections and Repair

The final answer is asked to contain these headings in order:

```text
**What Changed**
...

**User Impact**
...

**Security & Performance**
...
```

The requested fallback for no changes is `No changes found in the requested timeframe based on provided context.` The security/performance fallback is `None identified`.

`hasStandardSections` checks heading substrings in order; it does not verify section bodies, require exact heading lines, or reject a preamble. If the check fails, the pipeline makes one format-repair call using the existing answer, without re-retrieving evidence. It adopts the repair only if that check passes. A failed or still-invalid repair returns the original answer with a warning.

The standard prompt says the changes were stitched with neighbors, but the active scroll path does not do stitching. This is a prompt/source mismatch worth knowing when changing standard retrieval.

All retrieved changes being presented to batch calls does not guarantee that the model or reduction preserves every fact. Input coverage and generated-summary completeness are different properties.

## 9. Path C: Generate or Update Documentation

Owners: `DOCPipeline.Execute` in [rag-go/pkg/pipeline/ragPipeline.go](rag-go/pkg/pipeline/ragPipeline.go), then `LLMDocProcessor.Process` in [rag-go/pkg/pipeline/docWorkflow.go](rag-go/pkg/pipeline/docWorkflow.go).

```text
question -> profile + query vectors
									|
					+-------+--------+
					|                |
					v                v
		query code         query documents       concurrent Qdrant calls
					|                |
		optional rerank   optional rerank        separate pools, sequential helper calls
					|                |
		optional cap      optional cap
					|                |
					|          index + fit triage input
					|                |
					|          LLM triage (unless no docs)
					|                |
					|          select relevant doc chunks
					|                |
					+--------+-------+
									 |
									 v
					 LLM compose document
									 |
									 v
					 local decision + confidence
									 |
									 v
			 JSON document result inside Response.answer
```

The vectors are computed once and shared between both retrieval calls. Two goroutines, buffered result channels, and a wait group perform the code/doc queries. The pipeline waits for both and **fails if either query errors**; a successful empty document result is different from a document retrieval failure. It does not cancel the other query early when one fails.

### 9.1 Stage 1: Triage Existing Documentation

`indexDocChunks` gives nonempty retrieved doc strings contiguous labels `[chunk 0]`, `[chunk 1]`, and so on. These labels are positions in the triage input, not Qdrant's stored `chunk_index` payload values.

`runTriage` asks a model which documents answer the question and how completely they cover it. It sees documentation and the question, **not code**. It assesses relevance and coverage, not correctness against the implementation.

```json
{
	"topic": "Configure a service",
	"coverage": "partial",
	"relevant_chunks": [{"index": 0, "relevance": 0.9, "why": "Documents the setup"}],
	"missing_points": ["How to validate the deployment"],
	"reason": "Setup is covered but validation is absent"
}
```

If there are no documentation chunks, this model call is skipped and coverage becomes `none`.

`normalizeTriage` initializes missing arrays, drops invalid indices with warnings, clamps scores to 0..1, normalizes coverage, and changes `complete` to `partial` if missing points contradict it. `selectRelevantChunks` deduplicates verdict indices, drops scores below **0.5**, records source references and reasons, and warns about dropped/ignored docs. If none survive, `Process` forces coverage to `none`.

### 9.2 Handling Malformed Triage JSON

`completeAndRepairJSON` attempts local normalization before spending another model call:

```text
raw output
	-> valid JSON already?
	-> remove outer code fence
	-> extract balanced JSON from surrounding prose
	-> escape literal control characters inside strings / remove trailing commas
	-> if still invalid: one LLM JSON repair call
	-> normalize repair or fail
	-> json.Unmarshal into DocTriageResult
```

Syntactically valid JSON is not necessarily a valid triage object. A valid but incompatible value can pass syntax normalization and then fail struct decoding without that repair path. This is not strict JSON-schema validation.

### 9.3 Stage 2: Compose from Docs Plus Authoritative Code

`defaultDocProfile` currently hardcodes:

```text
kind:      kb_article
sections:  Summary -> Context -> Steps -> Validation -> Notes
tone:      technical and concise
audience:  users and operators of this product
```

If relevant docs survive, compose uses them as a baseline and compares them with the source evidence. If no docs survive, it writes a fresh document from source evidence. Source code is the requested authority; conflicting documented values are to be corrected, and unsupported details go into warnings. Code evidence is supplied independently of triage, subject to prompt fitting.

The model replies in a plain-text envelope:

```text
TITLE: Configure the service
CORRECTION: One old documented setting was replaced with the source-backed setting.
WARNING: An external deployment detail could not be confirmed.
BODY:
## Summary
... complete Markdown follows ...
```

This avoids requiring the model to JSON-escape an entire Markdown document containing quotes, newlines, and code fences. `parseComposeEnvelope` reads the labels case-insensitively and treats everything after `BODY:` as literal body text. Optional correction/warning labels may repeat. It normalizes accidental outer Markdown fences and can derive a missing/placeholder title from the first usable body line.

`validateComposeResult` rejects empty/placeholder titles or bodies. `runCompose` makes at most two attempts for unusable content, adding corrective instructions for the second. A vLLM transport/error response fails immediately rather than using that content retry. Required sections, factual accuracy, command safety, and citation correctness are prompt requirements, not exhaustively checked by this validator.

### 9.4 Stage 3: Decide Locally What the Document Needs

`DefaultDocDecisionEngine.Decide` uses this order:

| Condition | Status |
| --- | --- |
| Coverage is `none` | `new_document_required` |
| Otherwise, compose reported any corrections | `update_required` |
| Otherwise, coverage is `partial` | `update_required` |
| Otherwise | `no_changes_required` |

Even `no_changes_required` includes composed document text. The decision is a recommendation; there is no file write, Git commit, Qdrant upsert, or publishing operation in this workflow.

`ComputeConfidence` is a heuristic, **not a calibrated probability**:

```text
start                                      0.40
complete coverage                         +0.25
partial coverage (instead of complete)    +0.10
at least two relevance scores >= 0.75     +0.15
exactly one such score (instead of two)   +0.08
no corrections                           +0.10
no compose warnings                      +0.10
more than two missing points             -0.10
clamp final score to [0.00, 0.95]
```

The score uses normalized triage verdicts and compose warnings before all final warnings are combined. It is not a count of independently verified facts and does not account for every downstream truncation/attribution issue.

### 9.5 The Nested JSON Response

`Process` marshals `DocProcessOutput` to a JSON **string**, then stores that string as `Response.Answer`:

```text
HTTP response JSON
	answer: "{\"status\":\"update_required\", ...}"
	type: ""
	sources: {code_chunks_retrieved: N, doc_chunks_retrieved: M}
	meta: {...}

caller parses outer JSON
	-> JSON-parse answer again
			 -> status, coverage, confidence_overall, topic, title,
					body_markdown, corrections, missing_points, matched_docs,
					decision_reason, warnings
```

`matched_docs[].chunk_index` refers to the triage label, not the original stored chunk index. Source references come from `[source: ...]`; missing references become `unattributed chunk`.

## 10. Prompt Construction: Where Answer Behavior Is Controlled

The owner is [rag-go/pkg/pipeline/prompGen.go](rag-go/pkg/pipeline/prompGen.go). This is the first place to change answer audience, structure, grounding rules, or instructional style.

| Builder/helper | Actual role |
| --- | --- |
| `buildPrompt` | Standard: system + user messages containing change window and changes. Nonstandard: one user message with guidance, app profile, code evidence, and question. |
| `joinChunks` | Joins evidence with `\n---\n`, or inserts a no-data fallback. |
| `isInstructionalRequest` | Keyword-substring heuristic for how-to/command/setup requests. |
| `isSupportQuestion` | Keyword-substring heuristic for support/capability questions; takes precedence over instructional guidance in the direct-answer branch. |
| `buildStandardMergePrompt` | Merges partial summaries without intentionally dropping distinct facts. |
| `buildStandardFormatRepairPrompt` | Reformats an existing standard answer. |
| `buildDocTriagePrompt` | Requests coverage and relevance judgments for indexed documents. |
| `buildDocComposePrompt` | Requests user/operator Markdown with source authority and the configured document profile. |
| `buildDocRepairPrompt` | Requests JSON repair for malformed structured output. |

Ordinary answers are explicitly aimed at product users/operators. The prompt generally prohibits source patches and implementation-level edits. Sending a developer question to this endpoint does not automatically turn it into a code-maintenance assistant. To support that use case, change prompt policy deliberately and add a clear request mode rather than expecting retrieval changes alone to solve it.

Instruction/support detection uses `strings.Contains`, not a classifier model or word-boundary matching. Short substrings can produce surprising matches. The request's `type` does not select a separate family of nonstandard prompts: every nonstandard value reaches the same direct-answer builder.

The ordinary prompt still contains a change-context heading with a no-data fallback even though that path queries only code. Likewise, some comments/rules describe stronger properties than the reader guarantees. Treat prompt text as instructions sent to a model, not as proof of what retrieval did.

Retrieved content is interpolated into messages as text. Grounding rules reduce speculation but are not an explicit prompt-injection defense or an authorization mechanism. No conversational history is stored or replayed by these endpoints; each request builds fresh messages.

## 11. Context and Output Token Budgets

The owner is [rag-go/pkg/pipeline/tokenBudget.go](rag-go/pkg/pipeline/tokenBudget.go). A model window must accommodate both the input prompt and generated output:

```text
model context capacity
	= chat/system/query overhead + retrieved text + output tokens + safety margin
```

### 11.1 Current Constants and Approximation

| Setting | Value |
| --- | --- |
| Assumed model context | 32,768 tokens |
| Safety reserve | 2,048 tokens |
| General byte/token approximation | 4 |
| Standard diff byte/token approximation | 3, deliberately more conservative |
| Auto output budget | Clamped to 256..8,192 tokens |
| Explicit positive `token_limit` | Clamped to 64..12,288 tokens |
| Desired compose output room | 3,072 tokens |
| Desired triage output room | 1,024 tokens |
| Fixed prompt reserve for context cap | 2,000 tokens |
| Retrieved-text cap | `(32768 - 2048 - 3072 - 2000) * 4 = 102592` bytes approximately |

Names in the code say characters, but Go `len(string)` counts **bytes**. These estimates are not the model tokenizer, and non-ASCII text, code, and chat-template overhead can change the actual ratio. The assumed context size is hardcoded, not read from vLLM or an environment variable.

Without an explicit positive token limit, `ResolveTokenBudget` estimates input tokens, subtracts them and the safety margin from context capacity, and clamps the result. Clamping to a minimum does not make an oversized prompt fit: it can still produce a rejected request.

An explicit token limit is a hard output override after clamping. It does not mean total input-plus-output tokens and does not automatically shrink evidence to make room. Negative/zero values use automatic budgeting.

### 11.2 Three Different Fitting Strategies

**Current-code answer:** if `CONTEXT_TRUNCATION_ENABLED=true`, `TruncateChunksToCharBudget` retains whole chunks in existing order until the next would exceed the cap, then drops that chunk and everything after it. It never cuts a chunk in place and does not search later chunks for smaller ones. A first oversized chunk can leave no evidence. After this, evidence labels and prompts add extra text. There is no general automatic context-error retry on this path.

**Standard answer:** ignores this truncation switch for its change text. It packs/splits chunks into multiple calls and reduces the results, reserving output space and using the conservative diff estimate.

**Document generation:** optionally applies the 102,592-byte cap independently to code and doc pools first. That is not, by itself, a combined prompt cap. The processor then fits individual calls when automatic token budgeting is in use:

```text
fitTriageChunks:
	drop trailing indexed docs while more than one remains
	until the estimated triage output budget reaches 1024

fitComposePrompt:
	drop trailing docs while more than one remains
	then drop trailing code chunks, potentially all code
	until estimated compose output budget reaches 3072
	or nothing further can be removed by this policy
```

One doc baseline can remain even if it alone is too large. An explicit positive `token_limit` bypasses these processor fitting loops. Conversely, setting `CONTEXT_TRUNCATION_ENABLED=false` does **not** bypass the processor's automatic fitting loops. A small explicit output budget can also cause truncated/unusable document output.

`ResolveDocStepTokenBudget` uses 40% of the normal auto budget for triage and the full budget for compose/repair. Explicit overrides use the same clamped value for every step.

`AllocateChunkCharBudget` and `evidenceOnlyWeights` describe a weighted shared-pool utility, but the active `Execute` paths do not invoke it. Do not infer actual budget allocation from that utility or older comments alone.

### 11.3 What the Truncation Metadata Does Not Tell You

`context_truncation_active` records the initial pipeline chunk-cap removal only. It does not report later doc triage/compose fitting, relevance filtering, MMR selection, reranking, or source loss during model summarization. Standard responses leave it false even when batching splits text. Interpret it as a narrow implementation flag, not an end-to-end context completeness guarantee.

## 12. vLLM: Generating the Final Text

The client is [rag-go/pkg/vllm/clientVLLM.go](rag-go/pkg/vllm/clientVLLM.go). It is an HTTP OpenAI-compatible chat client, not a gRPC vLLM client in the current implementation.

```text
[]pipeline.Message + maxTokens
								|
								v
convert to chatMessage JSON objects
								|
								v
POST {VLLM base URL}/v1/chat/completions
								|
								v
decode choices[0].message.content
								|
								v
return one string to pipeline
```

Conceptual request body:

```json
{
	"model": "Qwen/Qwen2.5-7B-Instruct",
	"messages": [
		{"role": "system", "content": "Rules for this generation step"},
		{"role": "user", "content": "Question plus retrieved evidence"}
	],
	"temperature": 0,
	"top_p": 1,
	"seed": 42,
	"max_tokens": 4096
}
```

Actual messages and `max_tokens` vary by step; ordinary code answers use a single user message. Model selection comes from `QWEN_MODEL_NAME`. Temperature 0, top-p 1, and seed 42 are fixed in the client. These favor repeatability but do not guarantee identical answers across hardware, server versions, or concurrent execution.

The HTTP transport reuses connections, with 10 idle connections per host. That is an idle-pool setting, **not** a limit of 10 active network requests. `VLLM_TIMEOUT` applies to each HTTP completion call, not an entire multi-call standard/document pipeline.

Non-200 responses become errors; up to 2,048 bytes of the error body are logged and included in the internal error. Empty `choices` is an error. The client does not inspect finish reasons or token usage, validate a nonempty content string, attach API-key authorization, implement general retries, request constrained JSON decoding, or stream completion tokens. Higher pipeline layers provide the specific formatting/JSON/content repair behaviors described earlier.

## 13. Configuration and Deployment Reference

The source of defaults is [rag-go/main/config.go](rag-go/main/config.go). The table below lists **source defaults**, not necessarily the values in your deployment. Most malformed numeric/bool/duration strings fall back silently; configuration loading does not comprehensively validate ranges.

### 13.1 Server and Downstream Connections

| Environment variable | Source default | Purpose |
| --- | --- | --- |
| `EMBED_SERVICE_TYPE` | Required | `<e5 or gte-qwen>:<host[:port]>`; missing/malformed value prevents startup. |
| `FIBER_PORT` | `8080` | Listen port, normalized to `:8080`; use a port or colon-prefixed port, not an arbitrary host binding. |
| `FIBER_READ_TIMEOUT` | `30s` | HTTP server read timeout. |
| `FIBER_WRITE_TIMEOUT` | `120s` | HTTP server write timeout; consider the full workflow and SSE behavior. |
| `FIBER_IDLE_TIMEOUT` | `60s` | Idle HTTP connection timeout. |
| `EMBED_TIMEOUT` | `60s` | Each embed/sparse/rerank HTTP request timeout. |
| `QDRANT_HOST` | `qdrant-service:6334` | gRPC endpoint; port 6334 appended if no colon exists. |
| `QDRANT_MAX_CALL_RECV_MSG_KB` | `8192` | Maximum incoming gRPC response size, converted using 1 KB = 1024 bytes. Nonpositive input falls back to default. |
| `VLLM_HOST` | `qwen-3-service:80` | Chat server endpoint; default port 80 when absent. |
| `VLLM_TIMEOUT` | `120s` | Each completion HTTP call timeout. |
| `QWEN_MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Requested served model name. |
| `LOG_VERBOSE` | Not enabled | Exact value `true` selects debug logs. |

Host normalization is simple string logic, not a full URL/IPv6 parser. `buildHTTPURL` preserves a supplied HTTP/HTTPS prefix or adds `http://`; Qdrant expects a gRPC host target. Follow the `<host[:port]>` convention unless deliberately extending these helpers. Client code has no configured bearer-token support, and Qdrant uses insecure credentials; production TLS/authentication policy needs explicit handling or an appropriate trusted proxy/network boundary.

The Qdrant constructor has two attempts separated by 120 seconds **if `grpc.NewClient` itself errors**. gRPC client creation is lazy, so success there does not prove network connectivity or collection availability. Normal RPC failures do not use this constructor retry loop. If construction leaves `points` nil, request methods return `qdrant client not initialised`; there is no local reinitialization loop for that nil state. Do not read the startup log as a full dependency health check.

### 13.2 Retrieval and Context Controls

| Environment variable | Source default | Applies to |
| --- | --- | --- |
| `CODE_COLLECTION` | `code_chunks` | Current-code answers and doc generation. |
| `CHANGE_COLLECTION` | `change_chunks` | Standard change summaries. |
| `CHANGE_DATE_FIELD` | `date` | Standard scan's preferred date field, with the fallback field scans described earlier. |
| `DOC_COLLECTION` | `doc_chunks` | Document generation only. |
| `QDRANT_SCORE_THRESHOLD` | `0` | Positive values filter dense scores; 0/negative disables this filter. |
| `QDRANT_NEIGHBOR_STITCH` | `true` | Best-effort code/doc hit expansion. |
| `QDRANT_MMR_ENABLED` | `true` | Diversity selection and normalized-text deduplication on vector-query results. |
| `QDRANT_MMR_LAMBDA` | `0.7` | Relevance/diversity balance; values outside 0..1 reset to 0.7 in the constructor. |
| `QDRANT_MMR_OVERFETCH` | `3` | Candidate multiplier before MMR; constructor clamps values below 1 to 1. |
| `QDRANT_HYBRID_ENABLED` | `true` | Startup attachment of an optional sparse embedder. E5 remains dense-only. |
| `QDRANT_DENSE_VECTOR_NAME` | `dense` | Named dense vector; explicitly empty selects unnamed legacy vector. |
| `QDRANT_SPARSE_VECTOR_NAME` | `sparse` | Sparse vector name. The config helper replaces empty input with the default. |
| `QDRANT_HYBRID_PREFETCH_MULTIPLIER` | `2` | Candidate cap per dense/sparse prefetch; constructor clamps below 1 to 1. |
| `RERANK_ENABLED` | `false` | Cross-encoder reranking for code/doc vector results, not standard changes. |
| `RERANK_OVERFETCH_MULTIPLIER` | `4` | Pipeline candidate multiplier only when reranking is enabled and the multiplier exceeds 1. |
| `CONTEXT_TRUNCATION_ENABLED` | `true` | Initial whole-chunk caps for code/doc paths; not a switch for all processor fitting. |
| `APP_PROFILE_DIR` | `/etc/app-prof` | Directory for mounted product profiles. |

`AppProfileFiles` is a hardcoded repository-to-filename map in `loadConfig`, not an environment-provided map. Current entries are `github.com/Blazemeter/bzm-crane`, `taurus`, `helm-crane`, `bzm-mcp`, and `sv-mcp` under the same organization, with their corresponding `.txt` filenames. Adding a repository profile requires a mapping change and making its file available in the runtime directory.

### 13.3 Source Defaults vs Kubernetes Values

[container/rag-go-gte.yaml](container/rag-go-gte.yaml) is an example of deployment overrides:

```text
collection names:     gte-code_chunks / gte-change_chunks / gte-doc_chunks
gRPC receive cap:     16384 KB rather than source default 8192 KB
context truncation:   false rather than source default true
reranking:            false (configured multiplier 2 is inactive)
Fiber write timeout:  280s
vLLM call timeout:    380s
```

These values illustrate why reading only defaults can mislead debugging. A downstream call allowed 380 seconds does not guarantee that a caller, proxy, or server write timeout allows the whole workflow that long. Multiple generation rounds extend the overall latency further.

The manifest's service exposes port 80 and forwards to container port 8080. Its shown volume mount is `/tmp`, not an application-profile mount; a profile mapping alone does not make a profile file exist. Values such as `TRANSFORMERS_OFFLINE` in the Go deployment are not read by `loadConfig`; Python model settings belong on the embedding service deployment.

The E5 deployment is [container/rag-go-e5.yaml](container/rag-go-e5.yaml). Keep collection names and model choice aligned when comparing these deployments rather than switching only the client selector.

### 13.4 Building and Running

[rag-go/go.mod](rag-go/go.mod) declares Go `1.26.5`. Use a compatible toolchain; code includes APIs such as `sync.WaitGroup.Go`. Run a native local process from the module directory, with real/reachable dependencies and compatible existing collections:

```bash
cd rag-go
EMBED_SERVICE_TYPE='gte-qwen:localhost:8000' \
QDRANT_HOST='localhost:6334' \
VLLM_HOST='localhost:8001' \
go run ./main
```

The localhost ports in this command are example bindings you must arrange, not services automatically started by Orca. Change collection names and the model name if your services use different values.

[makefile](makefile) provides `build-binary`, `build-image`, `push-image`, and `release`. It requires `VERSION` for builds and targets Linux amd64 by default:

```bash
make build-binary VERSION=learning
```

That default binary is not a native macOS executable. To run locally on macOS, prefer the native `go run` command above or intentionally override build targets.

[container/Containerfile](container/Containerfile) builds a static Go binary in a Go 1.26 stage, injects `main.orcaVersion`, and copies it into a nonroot distroless image. It contains only the Go service; it does not embed Qdrant, the Python model service, or vLLM. The Docker build expects the module checksum file in its copy step, so ensure dependencies/checksums are available in the build context before using this route.

## 14. Worked Requests and Response Interpretation

The examples below assume Orca is reachable at `localhost:8080` and that the illustrative repository/component values have been replaced with values actually stored in your payloads.

### 14.1 Current-Code Question

```bash
curl -sS http://localhost:8080/api/v1/rag-go \
	-H 'Content-Type: application/json' \
	-d '{
		"query_text": "Does this application support a health check?",
		"repo_id": "github.com/example/project",
		"type": "code",
		"limit": 5,
		"token_limit": 2048
	}'
```

Trace it yourself:

```text
handleRAG
	-> validates scope/question; adds default dates (unused by this branch)
	-> Execute sees type != standard
	-> optional profile, dense + optional sparse question encoding
	-> code collection search, filters, fusion/MMR/stitch as configured
	-> optional reranking; optional initial context cap
	-> labels implementation/comment evidence
	-> support-question prompt guidance
	-> Complete(..., maxTokens=2048)
	-> outer response JSON
```

Illustrative response shape, not a captured live result:

```json
{
	"answer": "An answer based on the retrieved source evidence.",
	"type": "code",
	"sources": {
		"change_chunks_retrieved": 0,
		"code_chunks_retrieved": 5,
		"code_chunks_implementation": 2,
		"code_chunks_mixed": 1
	},
	"meta": {
		"repo_id": "github.com/example/project",
		"query_text": "Does this application support a health check?",
		"context_truncation_enabled": true,
		"context_truncation_active": true
	}
}
```

Here the example retrieval returned five strings but only three remained for evidence annotation after an initial context cap. Counts are intentionally different: `code_chunks_retrieved` is recorded immediately after `Query` returns, **after** its MMR/stitching but **before** pipeline reranking/truncation. Evidence counts describe the strings finally annotated. They are not raw Qdrant hit counts or counts of individual stitched neighbors. Kinds with no chunks may be absent rather than present with zero.

### 14.2 Standard Changes, Explicit Window, SSE Progress

```bash
curl -N 'http://localhost:8080/api/v1/rag-go?stream=true' \
	-H 'Content-Type: application/json' \
	-d '{
		"query_text": "Summarize changes and user impact.",
		"repo_id": "github.com/example/project",
		"type": "standard",
		"from_date": "2026-10-01",
		"to_date": "2026-10-22"
	}'
```

```text
request_received: 0
qdrant_query_complete: 25
context_ready: 30
messages_assembled: 55
generating_response: 60
batch_complete: ...                   only for multi-batch windows
vllm_complete: 95
complete: outer Response
```

There is no `embedding_complete` stage on this branch. `sources.change_batches` is the number of initial packed batches, not total model calls; recursive split calls, reduction rounds, and format repair add calls without increasing that initial count. `code_chunks_retrieved` is zero.

### 14.3 Document Generation

```bash
curl -sS http://localhost:8080/api/v1/rag-go/generate-doc \
	-H 'Content-Type: application/json' \
	-d '{
		"query_text": "How do I configure and validate the service?",
		"repo_id": "github.com/example/project",
		"component": "api",
		"limit": 25
	}'
```

Both scope fields must match when both are present. The route does not accept a separate document ID or overwrite an existing page. It finds relevant doc chunks semantically and returns a recommended status plus generated body.

For a JavaScript caller, parsing looks like:

```javascript
const outer = await response.json();
const documentResult = JSON.parse(outer.answer);
console.log(documentResult.status);
console.log(documentResult.body_markdown);
```

### 14.4 Compare the Branches at a Glance

| Property | Current-code answer | Standard changes | Document generation |
| --- | --- | --- | --- |
| Route | `/api/v1/rag-go` | `/api/v1/rag-go` | `/api/v1/rag-go/generate-doc` |
| Branch selector | Any nonstandard `type` | Normalized `type=standard` | Route itself |
| Collections | Code | Changes | Code + docs |
| Query embedding | Yes | No | Yes, once for both collections |
| Dates used | No | Yes, both bounds needed by retrieval | No |
| Meaning of `limit` | Primary context selection count | Does not cap the complete change scan | Per-collection primary selection count |
| RRF/MMR/stitching | As configured | No | As configured on both searches |
| Cross-encoder reranking | Optional | No | Optional, separately per collection |
| Profile loaded | Yes | No | Yes |
| Typical model calls | One | One per batch + reductions | Triage + compose, or just compose without docs |
| Extra model calls | None as a generic retry | Context split / merge / optional format repair | Optional JSON repair / compose content retry |
| Output in `answer` | Plain text | Sectioned summary | Stringified JSON object containing Markdown |
| Persistence | None | None | None |

## 15. Where to Make a Change

Use this section when describing a desired modification or deciding which code to open first.

| Desired change | Primary owner | Follow-through needed |
| --- | --- | --- |
| Add/rename an API field or change request defaults | [rag-go/pkg/handler/ragHandler.go](rag-go/pkg/handler/ragHandler.go): request structs and handler methods | Copy the value into `pipeline.Request`; update branch logic, examples, and handler tests. A JSON field declaration alone does not wire behavior. |
| Support `repo_name` as an actual selector | Same handler, then [rag-go/pkg/qdrant/clientQdrant.go](rag-go/pkg/qdrant/clientQdrant.go): `filterConditions` or a resolver before it | Choose a real name-to-ID mapping or backfill a payload field. Fix validation too; do not silently remove repository scope. |
| Add a new answer mode | [rag-go/pkg/pipeline/ragPipeline.go](rag-go/pkg/pipeline/ragPipeline.go): `Execute` | Define explicit mode validation/dispatch, retrieval policy, prompt, budgets, response expectations, and tests. Today unknown modes silently use code retrieval. |
| Search documentation for ordinary answers too | Same `RAGPipeline.Execute` and `QdrantQuerier` usage | Decide evidence authority and a combined prompt budget; do not just append an unbounded second pool. |
| Change startup dependencies or shared client wiring | [rag-go/main/main.go](rag-go/main/main.go) | Update configuration, interface compatibility, and deployment settings. |
| Add a new embedding provider | [rag-go/pkg/embedder/clientFactory.go](rag-go/pkg/embedder/clientFactory.go) plus a concrete client in the same package | Implement `Embed` and `Rerank` for the factory contract; add optional `SparseEmbed` if supported. Align stored vectors/model templates. |
| Change E5 formatting or model loading | [embed-service/e5-main.py](embed-service/e5-main.py) | Coordinate ingestion formatting and runtime model cache. |
| Change GTE templates/device/sparse encoding | [embed-service/gte-main.py](embed-service/gte-main.py) | Prefer existing env controls first; reindex when semantic or sparse compatibility changes. |
| Change collection payload keys, source extraction, filters | [rag-go/pkg/qdrant/clientQdrant.go](rag-go/pkg/qdrant/clientQdrant.go) | Coordinate the external ingester, existing records, payload indexes, both vector/scroll paths, and neighbor lookup. |
| Tune semantic/lexical candidate selection | Same file: `query`, prefetch setup, `selectDiverse` | Compare exact-identifier and semantic queries; keep dense thresholds separate from fused scores. |
| Change neighbor expansion | Same file: `stitchNeighborChunks`, `scrollByFilePath` | Test source attribution, index types, duplicate overlap, response sizes, and latency. |
| Change change-window behavior | Handler date defaults, then Qdrant `standardDateFilters` / `QueryStandardAll` | Test inclusive boundaries, one-sided inputs, fallback fields, pagination, deduplication, and UTC behavior. |
| Tune standard LLM parallelism or merge strategy | [rag-go/pkg/pipeline/standardParallel.go](rag-go/pkg/pipeline/standardParallel.go) | Consider per-request vs global limits and context-overflow behavior in both map and reduce. |
| Change user/operator voice to developer explanations | [rag-go/pkg/pipeline/prompGen.go](rag-go/pkg/pipeline/prompGen.go) | Change audience prohibitions intentionally; keep authority rules and regression questions. |
| Change document relevance, required sections, decision statuses | [rag-go/pkg/pipeline/docWorkflow.go](rag-go/pkg/pipeline/docWorkflow.go) | Preserve parser/output contracts; confidence and decision policy may need separate changes. |
| Change evidence classification | [rag-go/pkg/pipeline/evidence.go](rag-go/pkg/pipeline/evidence.go) | Test multiple languages; a heuristic change is not proof of execution or reachability. |
| Change output budget/context window | [rag-go/pkg/pipeline/tokenBudget.go](rag-go/pkg/pipeline/tokenBudget.go) | Match actual vLLM context, include prompt/output overhead, test every branch and explicit overrides. |
| Change model sampling, auth, request/response handling | [rag-go/pkg/vllm/clientVLLM.go](rag-go/pkg/vllm/clientVLLM.go) | Verify served-model/API support, timeout behavior, errors, and repeated-run quality. |
| Add real token streaming | vLLM client interface/implementation, pipelines, and SSE handler together | The current `Complete -> string` abstraction cannot deliver incremental tokens. Define progress vs token vs final events and cancellation. |
| Persist generated docs | Add an explicit persistence boundary after document processing | Define destination, IDs, update policy, approval/idempotency, and authorization. Do not treat a generated status as an actual write. |
| Add dependency readiness | Handler/registration plus client health contracts | Keep readiness distinct from process liveness; avoid costly model inference as a routine probe. |

### 15.1 A Reliable Way to Describe an Efficiency Change

Instead of "make RAG faster," describe the branch, observed stage, scope, constraint, and measurement:

```text
Branch: current-code answers, GTE hybrid retrieval.
Observation: reranking dominates p95 after neighbor stitching.
Goal: lower p95 without losing exact env-var matches.
Scope: candidate multipliers and rerank input, not standard summaries.
Evidence: Qdrant candidate counts, rerank timing, evaluation questions.
Likely owners: qdrant.query, rerankChunks, and config defaults/manifest.
Check: compare answer correctness, source recall, error rate, and p95.
```

That information points to a bounded change and a meaningful check rather than a speculative refactor.

## 16. Efficiency Tuning Without Guessing

Separate retrieval latency, model latency, and answer quality. Fewer chunks can reduce work while also removing the only evidence that supports the answer.

### 16.1 Cost Model by Branch

Approximate critical paths, excluding queueing and validation:

```text
current-code:
	profile read
	+ max(dense encoding, sparse encoding)
	+ code query and neighbor lookups
	+ optional reranking
	+ one completion

documents:
	profile read
	+ max(dense encoding, sparse encoding)
	+ max(code retrieval, doc retrieval)
	+ code reranking + doc reranking (if enabled)
	+ optional triage and repair
	+ compose and optional content retry

standard:
	all date-field scrolls/pages
	+ bounded parallel batch summary waves
	+ merge rounds
	+ optional format repair
```

Concurrent requests share actual CPU/GPU/network capacity, so `max(...)` indicates overlap in the orchestration, not a guaranteed latency bound under load.

### 16.2 Existing Knobs and Their Tradeoffs

| Lever | Potential benefit | Risk/cost |
| --- | --- | --- |
| Lower `limit` | Less context, reranking, and generation input | Misses relevant evidence; standard scanning is unaffected. |
| Lower rerank/MMR/hybrid multipliers | Smaller candidate pools and responses | Can reduce recall/diversity before a strong reranker sees useful candidates. |
| Enable cross-encoder reranking | Better query-specific ordering from an overfetched pool | Extra model call, often expensive for long stitched candidates. |
| Keep hybrid enabled for identifier-heavy questions | Exact names/config keys can survive semantic mismatch | Extra sparse HTTP call and Qdrant branch; requires compatible ingestion. |
| Disable neighbor stitching | Fewer scrolls and less text | Context can start/end mid-function or omit required adjacent settings. |
| Raise dense threshold | Removes weak dense matches | Can remove useful evidence; it does not filter sparse hybrid hits. |
| Tune MMR lambda | Adjusts relevance vs redundancy | Too much diversity can admit weakly relevant material. |
| Keep automatic context fitting | More room for complete output, fewer huge prompts | Low-ranked evidence can be dropped; current metadata is incomplete. |
| Lower output `token_limit` | Caps generation work per call | Can cut off a document or invalidly constrain structured output. |
| Narrow standard date range | Fewer scroll results and LLM batches | Changes reporting scope; must not silently replace full-window coverage. |
| Align device/model service capacity | Reduces embedding/rerank bottlenecks | Requires real device availability, memory budgeting, and deployment changes. |

Source defaults are not measured optimal settings. Change one family of controls at a time against a fixed evaluation set.

### 16.3 Targeted Code Improvements to Consider Later

These are possible improvements, **not features this documentation change implements**:

1. Add per-stage durations and counts: embedding, sparse encoding, Qdrant RPCs, neighbor lookup, rerank, triage, compose, merge, total request time, and token usage/finish reason when available.
2. Introduce shared admission/concurrency limits if multiple standard requests overload vLLM. The current per-request semaphore does not protect aggregate model capacity.
3. Cache query embeddings only with a bounded eviction policy and a key including model/version/templates. Profile caching similarly needs file-change invalidation. Neither cache exists now.
4. Batch or bound neighbor lookups if measurements identify them as a bottleneck. Keep scope filters and deterministic source attribution intact.
5. Parallelize the two document reranking pools only if the embedding/reranker server has spare capacity; concurrency can increase GPU contention rather than reduce latency.
6. Use model-aware token counts or server-compatible tokenizer estimates. Include app profiles, labels, repairs, actual output reservation, and explicit-limit behavior.
7. Add maximum request limits/date-window policies if exposing expensive endpoints. The HTTP handler currently has defaults but no explicit upper bound on `limit`, date span, or overall downstream call count.
8. Consider streaming/page-wise standard batch preparation to reduce peak memory for large windows, preserving deduplication and complete-window semantics. Current code accumulates all changes before batching.
9. Add structured provenance alongside text so scores, point IDs, original indices, and fitting decisions survive to the response. The current `[]string` contract discards much of that information.

Evaluate using semantic paraphrases, exact function/env-var names, unrelated questions, empty results, stale docs, large diffs, and multi-repository component filters. Measure correctness/source coverage alongside p50/p95 latency and failure rate. Do not count a faster unsupported answer as a successful optimization.

## 17. Debugging and Verification

### 17.1 Read Logs in the Same Order as the Request

```text
rag request received / doc generate/update request received
	-> application profile loaded (nonstandard/doc paths, if available)
	-> embedding received (debug)
	-> sparse embedding received (debug, if applicable)
	-> qdrant query complete / standard change scroll complete
	-> rerank warnings if fallback occurred
	-> truncation warning if initial chunks were dropped
	-> assembled vllm messages / assembled standard batch messages
	-> vllm http completion received
	-> optional repair/failure logs
```

The message hash (`messages_sha256`) fingerprints marshaled pipeline messages on selected RAG/standard calls. It helps compare prompt assembly without logging entire retrieved code, but it is not an answer cache key or trace ID. Token limits, served model, sampling configuration, and downstream scheduling are not all represented in that hash. There is no built-in request correlation ID or comprehensive timing breakdown in these functions.

Qdrant query logs include returned hit counts, raw Qdrant candidates, parsed candidates, candidates with returned vectors, MMR settings, hybrid status, sparse term count, and threshold. `LOG_VERBOSE=true` also emits per-hit IDs/scores/order. This is often the most useful first evidence when retrieval quality changes.

### 17.2 Failure Policy

| Situation | Behavior |
| --- | --- |
| Missing/malformed required embed selector | Startup fails. |
| Dense embedding fails | Request fails; no retrieval/answer is generated. |
| Sparse embedding fails | Warning; continue dense-only. |
| Qdrant code/change/doc query fails | Request fails before generation; no answer from partial retrieval. |
| Neighbor lookup fails | Warning; retain the original hit text. |
| Cross-encoder fails or returns wrong count | Warning; keep original retrieval order up to requested count. |
| Missing/unreadable application profile | Warning/no profile; continue. |
| No retrieval matches, but query itself succeeds | Continue with no-data context; this is not a transport error. |
| vLLM ordinary answer call fails | Request fails. |
| Standard batch context overflow | Split/retry if there are at least two pieces; otherwise fail. |
| Standard merge call fails | Request fails. |
| Standard format repair fails | Return original answer. |
| Triage output has malformed JSON | Local normalization, then at most one model repair; fail if still unusable. |
| Compose content is empty/placeholder | At most one additional compose attempt. |
| Oversized Qdrant gRPC response | Special user-facing error states no answer was generated. |

### 17.3 Symptoms to Code Locations

| Symptom | First checks |
| --- | --- |
| Wrong repository or unexpectedly broad results | Actual payload `repo_id`/`component`; component-only scope; ignored `repo_name`; handler-to-request copying. |
| Exact identifier not found | GTE sparse availability, hybrid startup setting, sparse model compatibility, vector names, and filtering. |
| Weak semantic matches after model switch | Ingestion/query model pair, dimensions, query/document templates, correct collection name. |
| Repeated evidence | MMR state, normalized-text deduplication, neighbor overlap, duplicate ingestion points. |
| Correct chunk retrieved but missing from answer | Reranking cutoff, whole-chunk cap, doc fitting, model prompt policy, and generated-summary loss. |
| One-sided standard date request returns 500 | Handler accepts it but `standardDateFilters` needs both dates. |
| Standard output misses expected changes | Date fields/types and scope, full scroll success, batching/merge prompts; `limit` is not the scan cap. |
| Docs not used despite retrieval hits | Triage indices/scores, 0.5 relevance floor, coverage normalization, fitting. |
| Doc status unexpectedly says no changes | Decision engine only sees coverage and model-reported corrections, not an independent factual audit. |
| Empty/truncated document | Output budget, finish reason not inspected by client, compose validation, prompt size. |
| Context error with truncation off | Expected: no unlimited model window; doc fitting still applies unless explicit token limit bypasses it. |
| `context_truncation_active=false` but evidence missing | Later doc fitting/filtering and other selection stages are not captured by that flag. |
| Health returns OK while requests fail | `/health` is process liveness only. Probe dependencies separately. |
| Qdrant `received message larger than max` | Response payload size, returned MMR vectors, page size/candidate caps, and `QDRANT_MAX_CALL_RECV_MSG_KB`. Raising the cap is not a model-context fix. |
| Long standard requests fail near a consistent time | Server/proxy/client deadlines versus many per-call vLLM timeouts and merge rounds. |

### 17.4 Existing Tests and Their Scope

[rag-go/pkg/pipeline/ragPipeline_test.go](rag-go/pkg/pipeline/ragPipeline_test.go) checks that RAG and document retrieval failures propagate and stop model generation. It uses mock Qdrant/embedding/LLM implementations, so it does not require a running model or database.

[rag-go/pkg/handler/ragHandler_test.go](rag-go/pkg/handler/ragHandler_test.go) checks error delivery through normal JSON and SSE responses, including the specialized oversized-response message and avoiding exposure of private downstream details.

Useful commands from the module directory:

```bash
cd rag-go
go test ./pkg/handler ./pkg/pipeline
go test -race ./pkg/handler ./pkg/pipeline
go test ./...
go build ./main
```

These are suggested verification commands for future changes, not a claim that they were executed for this documentation-only task. The existing tests do not establish successful hybrid retrieval, date filtering, MMR quality, prompt fitting, JSON repair, or end-to-end model correctness.

For a change, add focused cases in the owning package, using existing tests as style examples. Good discriminating cases include:

- API field forwarding and rejected/normalized request modes.
- Standard inclusive day boundaries and missing bound behavior.
- Sparse outage fallback versus dense failure stopping execution.
- Qdrant named/unnamed vector request shape, hybrid filters, and threshold placement.
- MMR repeated text and missing-vector behavior; neighbor expansion with integer indices.
- Rerank error/count mismatch/tied-score fallback and ordering.
- Explicit token overrides, oversized first chunks, and doc fitting that preserves only one baseline.
- Triage JSON repairs, invalid/out-of-range indices, low relevance, and contradictory coverage.
- Compose envelope parsing, placeholder rejection, all decision statuses, and nested response serialization.

For model-quality changes, deterministic unit tests need a separate representative live evaluation. Unit tests can prove control flow and serialization, not that a new prompt or model produces better answers.

## 18. Current Boundaries and Caveats

Keep these boundaries visible when planning work:

1. **No Go ingestion or extraction pipeline.** Stored vectors/chunks must come from elsewhere. The GTE Python document-embedding endpoint helps external ingestion but is not a database writer.
2. **No document persistence.** Generated Markdown and statuses are returned, not published or indexed automatically.
3. **No multi-turn memory.** Requests are standalone; profiles provide product context rather than conversation history.
4. **No repo-name resolution.** The field exists but is not used. Component-only requests can cross repository boundaries.
5. **No strict answer verification.** Evidence labels, prompts, document decisions, and confidence scores depend on heuristics/model output, not a formal audit of claims.
6. **No unlimited context.** Disabling the initial cap does not expand vLLM's window. Explicit token overrides can bypass helpful doc fitting.
7. **No general retry strategy.** Only particular construction/format/content/context paths retry as described; arbitrary service outages do not get generic retries.
8. **No token-level streaming or complete disconnect handling.** SSE is milestone streaming followed by the complete response.
9. **No built-in application authentication/authorization layer in these routes.** Network placement, external access controls, transport security, and model credentials need explicit operational design.
10. **No global workload limiter.** Per-request standard concurrency and connection-pool idle settings are not aggregate capacity controls.

The most useful ownership model to remember is:

```text
handler     decides what input is accepted and how output is transported
pipeline    decides which workflow runs and in what order
embedder    converts the question to query vectors and scores candidate text
qdrant      decides how records are filtered, searched, diversified, and expanded
prompt code decides what the model is asked to do with those records
budget code decides how much text/output each call can attempt
vllm client transports chat messages and extracts generated content
```

When changing behavior, identify which of these layers actually makes the decision, then verify the smallest observable case that distinguishes the old behavior from the intended new behavior.
