package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/rs/zerolog"
)

// Interfaces — swap real clients for mocks in tests.
type QdrantQuerier interface {
	Query(ctx context.Context, collection string, vector []float32, sparseIndices []uint32, sparseValues []float32, repoID, component string, limit int) ([]string, error)
	QueryStandard(ctx context.Context, collection string, vector []float32, sparseIndices []uint32, sparseValues []float32, repoID, component string, limit int, fromDate, toDate, dateField string) ([]string, error)
	QueryStandardAll(ctx context.Context, collection string, repoID, component string, fromDate, toDate, dateField string) ([]string, error)
}

type VLLMCompleter interface {
	Complete(ctx context.Context, messages []Message, maxTokens int) (string, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// SparseEmbedder encodes query text into the BM25 sparse vector used for hybrid retrieval.
type SparseEmbedder interface {
	SparseEmbed(ctx context.Context, text string) ([]uint32, []float32, error)
}

// Reranker scores (query, document) pairs with a cross-encoder, higher is more relevant.
type Reranker interface {
	Rerank(ctx context.Context, query string, documents []string) ([]float32, error)
}

// Message is a minimal chat message passed to the LLM.
type Message struct {
	Role    string
	Content string
}

// ProgressEvent describes a completed pipeline milestone for one request.
type ProgressEvent struct {
	Stage   string `json:"stage"`
	Message string `json:"message"`
	Percent int    `json:"percent"`
}

// ProgressReporter receives request-scoped pipeline milestones.
type ProgressReporter func(ProgressEvent)

// Request is the pipeline's input — decoupled from the HTTP layer.
type Request struct {
	QueryText  string
	RepoID     string
	AppProfile string
	Type       string
	Limit      int
	TokenLimit int
	RepoName   string
	Component  string
	FromDate   string // YYYY-MM-DD, optional; filters chunks to this date or later.
	ToDate     string // YYYY-MM-DD, optional; filters chunks to this date or earlier.
	Progress   ProgressReporter
}

// ReportProgress sends a milestone only when the caller requested progress updates.
func (r Request) ReportProgress(stage, message string, percent int) {
	if r.Progress != nil {
		r.Progress(ProgressEvent{Stage: stage, Message: message, Percent: percent})
	}
}

// Response is what the pipeline returns to the handler.
type Response struct {
	Answer  string         `json:"answer"`
	Type    string         `json:"type"`
	Sources map[string]int `json:"sources"`
	Meta    ResponseMeta   `json:"meta"`
}

type ResponseMeta struct {
	RepoID                   string `json:"repo_id"`
	Component                string `json:"component,omitempty"`
	QueryText                string `json:"query_text"`
	ContextTruncationEnabled bool   `json:"context_truncation_enabled"`
	ContextTruncationActive  bool   `json:"context_truncation_active"`
}

// RAGPipeline wires all downstream clients together.
type RAGPipeline struct {
	qdrant                    QdrantQuerier
	vllm                      VLLMCompleter
	embedder                  Embedder
	sparseEmbedder            SparseEmbedder
	reranker                  Reranker
	rerankEnabled             bool
	rerankOverfetchMultiplier int
	contextTruncationEnabled  bool
	changeCollection          string
	codeCollection            string
	changeDateField           string
	appProfileDir             string
	appProfileFiles           map[string]string
	log                       zerolog.Logger
}

// RAGPipeline wires all downstream clients together.
type DOCPipeline struct {
	qdrant                    QdrantQuerier
	vllm                      VLLMCompleter
	embedder                  Embedder
	sparseEmbedder            SparseEmbedder
	reranker                  Reranker
	rerankEnabled             bool
	rerankOverfetchMultiplier int
	docProcessor              DocProcessor
	codeCollection            string
	docCollection             string
	contextTruncationEnabled  bool
	appProfileDir             string
	appProfileFiles           map[string]string
	log                       zerolog.Logger
}

func New(
	qdrant QdrantQuerier,
	vllm VLLMCompleter,
	embedder Embedder,
	reranker Reranker,
	rerankEnabled bool,
	rerankOverfetchMultiplier int,
	contextTruncationEnabled bool,
	changeCollection string,
	codeCollection string,
	changeDateField string,
	appProfileDir string,
	appProfileFiles map[string]string,
) *RAGPipeline {
	return &RAGPipeline{
		qdrant:                    qdrant,
		vllm:                      vllm,
		embedder:                  embedder,
		reranker:                  reranker,
		rerankEnabled:             rerankEnabled,
		rerankOverfetchMultiplier: rerankOverfetchMultiplier,
		contextTruncationEnabled:  contextTruncationEnabled,
		changeCollection:          changeCollection,
		codeCollection:            codeCollection,
		changeDateField:           changeDateField,
		appProfileDir:             appProfileDir,
		appProfileFiles:           appProfileFiles,
		log:                       zerolog.Nop(),
	}
}

func NewDoc(
	qdrant QdrantQuerier,
	vllm VLLMCompleter,
	embedder Embedder,
	reranker Reranker,
	rerankEnabled bool,
	rerankOverfetchMultiplier int,
	contextTruncationEnabled bool,
	codeCollection string,
	docCollection string,
	appProfileDir string,
	appProfileFiles map[string]string,
) *DOCPipeline {
	docProcessor := NewLLMDocProcessor(vllm, NewDefaultDocDecisionEngine())

	return &DOCPipeline{
		qdrant:                    qdrant,
		vllm:                      vllm,
		embedder:                  embedder,
		reranker:                  reranker,
		rerankEnabled:             rerankEnabled,
		rerankOverfetchMultiplier: rerankOverfetchMultiplier,
		docProcessor:              docProcessor,
		codeCollection:            codeCollection,
		docCollection:             docCollection,
		contextTruncationEnabled:  contextTruncationEnabled,
		appProfileDir:             appProfileDir,
		appProfileFiles:           appProfileFiles,
		log:                       zerolog.Nop(),
	}
}

func (p *RAGPipeline) WithLogger(log zerolog.Logger) *RAGPipeline {
	p.log = log
	return p
}

func (p *DOCPipeline) WithLogger(log zerolog.Logger) *DOCPipeline {
	p.log = log
	return p
}

// WithSparseEmbedder enables hybrid dense+sparse retrieval; nil keeps dense-only retrieval.
func (p *RAGPipeline) WithSparseEmbedder(sparse SparseEmbedder) *RAGPipeline {
	p.sparseEmbedder = sparse
	return p
}

func (p *DOCPipeline) WithSparseEmbedder(sparse SparseEmbedder) *DOCPipeline {
	p.sparseEmbedder = sparse
	return p
}

// queryVectors is the encoded retrieval query; empty sparse fields mean dense-only retrieval.
type queryVectors struct {
	dense         []float32
	sparseIndices []uint32
	sparseValues  []float32
}

// embedQuery encodes dense and sparse vectors concurrently. A sparse failure only
// degrades retrieval to dense-only; a dense failure fails the request.
func embedQuery(ctx context.Context, embedder Embedder, sparse SparseEmbedder, text string, log zerolog.Logger) (queryVectors, error) {
	var vectors queryVectors
	var sparseErr error
	var wg sync.WaitGroup
	if sparse != nil {
		wg.Add(1)
		go func() {
			defer wg.Done()
			vectors.sparseIndices, vectors.sparseValues, sparseErr = sparse.SparseEmbed(ctx, text)
		}()
	}
	dense, err := embedder.Embed(ctx, text)
	wg.Wait()
	if err != nil {
		return queryVectors{}, err
	}
	vectors.dense = dense
	if sparseErr != nil {
		log.Warn().Err(sparseErr).Msg("sparse query encoding failed, falling back to dense-only retrieval")
		vectors.sparseIndices, vectors.sparseValues = nil, nil
	}
	return vectors, nil
}

// Execute runs the full RAG pipeline for a single request. Standard summarizes what changed,
// so it retrieves change_chunks only; every other request answers from the current
// implementation, so it retrieves code_chunks only.
func (p *RAGPipeline) Execute(ctx context.Context, req Request) (*Response, error) {
	isStandard := strings.EqualFold(strings.TrimSpace(req.Type), "standard")

	var changeChunks, codeChunks []string
	var retrievedChangeCount, retrievedCodeCount int
	var contextTruncationActive bool

	if isStandard {
		chunks, err := p.qdrant.QueryStandardAll(ctx, p.changeCollection, req.RepoID, req.Component, req.FromDate, req.ToDate, p.changeDateField)
		if err != nil {
			p.log.Warn().Err(err).Str("collection", p.changeCollection).Msg("qdrant query failed")
		}
		retrievedChangeCount = len(chunks)
		changeChunks = chunks
		req.ReportProgress("qdrant_query_complete", "Qdrant query complete", 25)
		req.ReportProgress("context_ready", "Standard change context ready", 30)
	} else {
		req.AppProfile = resolveAppProfile(p.appProfileDir, p.appProfileFiles, req.RepoID, p.log)

		vectors, err := embedQuery(ctx, p.embedder, p.sparseEmbedder, req.QueryText, p.log)
		if err != nil {
			return nil, fmt.Errorf("embed: %w", err)
		}
		req.ReportProgress("embedding_complete", "Embedding complete", 15)

		// Over-fetch a larger candidate pool when reranking is enabled, so the cross-encoder
		// has more to choose from.
		queryLimit := req.Limit
		if p.rerankEnabled && p.rerankOverfetchMultiplier > 1 {
			queryLimit = req.Limit * p.rerankOverfetchMultiplier
		}

		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vectors.dense, vectors.sparseIndices, vectors.sparseValues, req.RepoID, req.Component, queryLimit)
		if err != nil {
			p.log.Warn().Err(err).Str("collection", p.codeCollection).Msg("qdrant query failed")
		}
		retrievedCodeCount = len(chunks)
		req.ReportProgress("qdrant_query_complete", "Qdrant query complete", 25)

		if p.rerankEnabled {
			chunks = rerankChunks(ctx, p.reranker, req.QueryText, chunks, req.Limit, p.log)
			req.ReportProgress("re_ranking", "Re-ranking complete", 35)
		}
		if p.contextTruncationEnabled {
			truncated := TruncateChunksToCharBudget(chunks, maxContextCharsTotal)
			contextTruncationActive = len(truncated) < len(chunks)
			chunks = truncated
		}
		if contextTruncationActive {
			p.log.Warn().
				Int("code_chunks_kept", len(chunks)).
				Int("code_chunks_retrieved", retrievedCodeCount).
				Msg("truncated retrieved chunks to stay within context budget")
		}
		codeChunks = chunks
	}

	codeChunks, evidenceCounts := annotateCodeChunks(codeChunks)
	var answer string
	var changeBatches int
	if isStandard {
		var err error
		answer, changeBatches, err = p.answerStandard(ctx, req, changeChunks)
		if err != nil {
			return nil, err
		}
	} else {
		messages := buildPrompt(req, changeChunks, codeChunks)
		maxTokens := ResolveTokenBudget(req, messages)
		req.ReportProgress("messages_assembled", "Assembled vLLM messages", 55)
		req.ReportProgress("generating_response", "Generating LLM response...", 60)
		// 4. Call LLM
		p.log.Info().
			Str("messages_sha256", hashMessages(messages)).
			Int("message_count", len(messages)).
			Int("max_tokens", maxTokens).
			Interface("code_evidence", evidenceCounts).
			Msg("assembled vllm messages")
		var err error
		answer, err = p.vllm.Complete(ctx, messages, maxTokens)
		if err != nil {
			return nil, fmt.Errorf("vllm complete: %w", err)
		}
	}
	req.ReportProgress("vllm_complete", "vLLM completion complete", 95)

	// 5. Enforce the section template for standard answers, with one reformat retry.
	if isStandard && !hasStandardSections(answer) {
		p.log.Warn().Msg("standard answer missing required sections, attempting reformat")
		repairPrompt := buildStandardFormatRepairPrompt(answer)
		repaired, repairErr := p.vllm.Complete(ctx, repairPrompt, ResolveTokenBudget(req, repairPrompt))
		switch {
		case repairErr != nil:
			p.log.Warn().Err(repairErr).Msg("standard answer reformat failed, returning original")
		case hasStandardSections(repaired):
			answer = repaired
			req.ReportProgress("vllm_complete", "vLLM format repair complete", 95)
		default:
			p.log.Warn().Msg("standard answer reformat still missing sections, returning original")
		}
	}

	sources := map[string]int{
		"change_chunks_retrieved": retrievedChangeCount,
		"code_chunks_retrieved":   retrievedCodeCount,
	}
	for kind, n := range evidenceCounts {
		sources["code_chunks_"+strings.ReplaceAll(kind, "-", "_")] = n
	}
	if isStandard {
		sources["change_batches"] = changeBatches
	}

	return &Response{
		Answer:  answer,
		Type:    req.Type,
		Sources: sources,
		Meta: ResponseMeta{
			RepoID:                   req.RepoID,
			Component:                req.Component,
			QueryText:                req.QueryText,
			ContextTruncationEnabled: p.contextTruncationEnabled,
			ContextTruncationActive:  contextTruncationActive,
		},
	}, nil
}

// answerStandard summarizes every change chunk without truncation. When the change set does
// not fit one context window it is split into batches (map), each summarized with the normal
// standard prompt, and the partial summaries are merged into one answer (reduce).
func (p *RAGPipeline) answerStandard(ctx context.Context, req Request, changeChunks []string) (string, int, error) {
	batches := packChunksByTokens(changeChunks, standardChunkTokenBudget(req, buildPrompt(req, nil, nil)))
	req.ReportProgress("messages_assembled", "Assembled vLLM messages", 55)
	req.ReportProgress("generating_response", "Generating LLM response...", 60)
	if len(batches) > 1 {
		p.log.Info().
			Int("change_chunks", len(changeChunks)).
			Int("batches", len(batches)).
			Msg("standard change set exceeds context window, summarizing in batches")
	}

	// The SSE progress writer is not goroutine-safe.
	var progressMu sync.Mutex
	done := 0
	results := make([][]string, len(batches))
	err := runParallel(ctx, len(batches), func(ctx context.Context, i int) error {
		out, err := p.summarizeStandardBatch(ctx, req, batches[i])
		if err != nil {
			return err
		}
		results[i] = out
		if len(batches) > 1 {
			progressMu.Lock()
			done++
			req.ReportProgress("batch_complete", fmt.Sprintf("Summarized change batch %d of %d", done, len(batches)), 60+25*done/len(batches))
			progressMu.Unlock()
		}
		return nil
	})
	if err != nil {
		return "", len(batches), err
	}

	var partials []string
	for _, out := range results {
		partials = append(partials, out...)
	}

	answer, err := p.mergeStandardPartials(ctx, req, partials)
	return answer, len(batches), err
}

// summarizeStandardBatch runs the standard prompt over one batch. If vLLM still rejects it for
// context length (the estimate is approximate), the batch is halved and retried.
func (p *RAGPipeline) summarizeStandardBatch(ctx context.Context, req Request, chunks []string) ([]string, error) {
	messages := buildPrompt(req, chunks, nil)
	maxTokens := resolveStandardTokenBudget(req, messages)
	p.log.Info().
		Str("messages_sha256", hashMessages(messages)).
		Int("change_chunks", len(chunks)).
		Int("max_tokens", maxTokens).
		Msg("assembled standard batch messages")

	answer, err := p.vllm.Complete(ctx, messages, maxTokens)
	if err == nil {
		return []string{answer}, nil
	}
	if !isContextLengthError(err) || len(chunks) < 2 {
		return nil, fmt.Errorf("vllm complete: %w", err)
	}

	p.log.Warn().Int("change_chunks", len(chunks)).Msg("standard batch exceeded context window, splitting in half")
	mid := len(chunks) / 2
	left, err := p.summarizeStandardBatch(ctx, req, chunks[:mid])
	if err != nil {
		return nil, err
	}
	right, err := p.summarizeStandardBatch(ctx, req, chunks[mid:])
	if err != nil {
		return nil, err
	}
	return append(left, right...), nil
}

// mergeStandardPartials reduces partial summaries to one, merging in rounds when the
// partials themselves do not fit a single merge prompt.
func (p *RAGPipeline) mergeStandardPartials(ctx context.Context, req Request, partials []string) (string, error) {
	budget := standardChunkTokenBudget(req, buildStandardMergePrompt(req, nil))
	for len(partials) > 1 {
		groups := packChunksByTokens(partials, budget)
		if len(groups) >= len(partials) {
			// Each partial alone fills the budget; merge pairwise to guarantee progress.
			groups = groups[:0]
			for i := 0; i < len(partials); i += 2 {
				groups = append(groups, partials[i:min(i+2, len(partials))])
			}
		}

		merged := make([]string, len(groups))
		err := runParallel(ctx, len(groups), func(ctx context.Context, i int) error {
			group := groups[i]
			if len(group) == 1 {
				merged[i] = group[0]
				return nil
			}
			messages := buildStandardMergePrompt(req, group)
			maxTokens := resolveStandardTokenBudget(req, messages)
			p.log.Info().Int("partials", len(group)).Int("max_tokens", maxTokens).Msg("merging standard partial summaries")
			out, err := p.vllm.Complete(ctx, messages, maxTokens)
			if err != nil {
				return fmt.Errorf("vllm merge: %w", err)
			}
			merged[i] = out
			return nil
		})
		if err != nil {
			return "", err
		}
		partials = merged
	}
	if len(partials) == 0 {
		return "", nil
	}
	return partials[0], nil
}

// maxParallelLLMCalls caps concurrent vLLM requests per standard query so one large
// window cannot monopolize the model server.
const maxParallelLLMCalls = 4

// runParallel runs fn for indices [0, n) with at most maxParallelLLMCalls in flight.
// The first error cancels the remaining calls and is returned.
func runParallel(ctx context.Context, n int, fn func(ctx context.Context, i int) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	var once sync.Once
	var firstErr error
	sem := make(chan struct{}, maxParallelLLMCalls)
	for i := range n {
		sem <- struct{}{}
		if ctx.Err() != nil {
			<-sem
			break
		}
		wg.Go(func() {
			defer func() { <-sem }()
			if err := fn(ctx, i); err != nil {
				once.Do(func() {
					firstErr = err
					cancel()
				})
			}
		})
	}
	wg.Wait()
	if firstErr == nil && ctx.Err() != nil {
		return ctx.Err()
	}
	return firstErr
}

func isContextLengthError(err error) bool {
	return strings.Contains(strings.ToLower(err.Error()), "maximum context length")
}

// resolveAppProfile loads the product profile for a repo, returning "" when none is configured.
func resolveAppProfile(dir string, files map[string]string, repoID string, log zerolog.Logger) string {
	profileFile := ""
	if files != nil {
		profileFile = files[repoID]
	}

	profile, err := loadApplicationProfile(dir, profileFile)
	if err != nil {
		log.Warn().Err(err).Str("repo_id", repoID).Str("profile_file", profileFile).Msg("application profile lookup failed")
		return ""
	}
	if profile == "" {
		log.Warn().Str("repo_id", repoID).Str("profile_file", profileFile).Msg("no application profile loaded")
		return ""
	}

	log.Info().Str("repo_id", repoID).Str("profile_file", profileFile).Msg("application profile loaded")
	return profile
}

func hashMessages(messages []Message) string {
	b, err := json.Marshal(messages)
	if err != nil {
		return "marshal-error"
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func (p *DOCPipeline) Execute(ctx context.Context, req Request) (*Response, error) {
	req.AppProfile = resolveAppProfile(p.appProfileDir, p.appProfileFiles, req.RepoID, p.log)

	// 1. Embed the query once
	vectors, err := embedQuery(ctx, p.embedder, p.sparseEmbedder, req.QueryText, p.log)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	req.ReportProgress("embedding_complete", "Embedding complete", 15)

	// 2. Fan-out: query code and doc collections concurrently. Doc-gen answers from the
	// current implementation, so it no longer needs change history.
	queryLimit := req.Limit
	if p.rerankEnabled && p.rerankOverfetchMultiplier > 1 {
		queryLimit = req.Limit * p.rerankOverfetchMultiplier
	}

	type result struct {
		chunks []string
		err    error
	}

	var wg sync.WaitGroup
	codeCh := make(chan result, 1)
	docCh := make(chan result, 1)
	wg.Add(2)
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vectors.dense, vectors.sparseIndices, vectors.sparseValues, req.RepoID, req.Component, queryLimit)
		codeCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.docCollection, vectors.dense, vectors.sparseIndices, vectors.sparseValues, req.RepoID, req.Component, queryLimit)
		docCh <- result{chunks, err}
	}()
	wg.Wait()

	codeResult := <-codeCh
	docResult := <-docCh

	if codeResult.err != nil {
		p.log.Warn().Err(codeResult.err).Str("collection", p.codeCollection).Msg("qdrant query failed")
	}
	if docResult.err != nil {
		p.log.Warn().Err(docResult.err).Str("collection", p.docCollection).Msg("qdrant query failed")
	}

	retrievedCodeCount := len(codeResult.chunks)
	retrievedDocCount := len(docResult.chunks)
	req.ReportProgress("qdrant_query_complete", "Qdrant queries complete", 25)

	// 3. Rerank each evidence pool before context budgeting so every source type
	// contributes its most query-relevant chunks to the document workflow.
	if p.rerankEnabled {
		codeResult.chunks = rerankChunks(ctx, p.reranker, req.QueryText, codeResult.chunks, req.Limit, p.log)
		docResult.chunks = rerankChunks(ctx, p.reranker, req.QueryText, docResult.chunks, req.Limit, p.log)
	}

	// 4. Cap each side against the shared pool as a backstop; compose fitting still sheds
	// documentation before source evidence when both need to shrink further.
	codeChunks := codeResult.chunks
	docChunks := docResult.chunks
	if p.contextTruncationEnabled {
		codeChunks = TruncateChunksToCharBudget(codeResult.chunks, maxContextCharsTotal)
		docChunks = TruncateChunksToCharBudget(docResult.chunks, maxContextCharsTotal)
	}
	contextTruncationActive := len(codeChunks) < len(codeResult.chunks) ||
		len(docChunks) < len(docResult.chunks)
	if contextTruncationActive {
		p.log.Warn().
			Int("code_chunks_kept", len(codeChunks)).
			Int("code_chunks_retrieved", len(codeResult.chunks)).
			Int("doc_chunks_kept", len(docChunks)).
			Int("doc_chunks_retrieved", len(docResult.chunks)).
			Msg("truncated retrieved chunks to stay within context budget")
	}

	// 5. Run strong-confidence doc workflow (triage -> compose -> decide)
	answer, err := p.docProcessor.Process(ctx, req, codeChunks, docChunks)
	if err != nil {
		return nil, fmt.Errorf("doc workflow: %w", err)
	}

	return &Response{
		Answer: answer,
		Sources: map[string]int{
			"code_chunks_retrieved": retrievedCodeCount,
			"doc_chunks_retrieved":  retrievedDocCount,
		},
		Meta: ResponseMeta{
			RepoID:                   req.RepoID,
			Component:                req.Component,
			QueryText:                req.QueryText,
			ContextTruncationEnabled: p.contextTruncationEnabled,
			ContextTruncationActive:  contextTruncationActive,
		},
	}, nil
}
