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
	Query(ctx context.Context, collection string, vector []float32, repoID, component string, limit int) ([]string, error)
	QueryStandard(ctx context.Context, collection string, vector []float32, repoID, component string, limit int, fromDate, toDate, dateField string) ([]string, error)
	QueryStandardAll(ctx context.Context, collection string, repoID, component string, fromDate, toDate, dateField string) ([]string, error)
}

type VLLMCompleter interface {
	Complete(ctx context.Context, messages []Message, maxTokens int) (string, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
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
	} else {
		req.AppProfile = resolveAppProfile(p.appProfileDir, p.appProfileFiles, req.RepoID, p.log)

		vector, err := p.embedder.Embed(ctx, req.QueryText)
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

		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vector, req.RepoID, req.Component, queryLimit)
		if err != nil {
			p.log.Warn().Err(err).Str("collection", p.codeCollection).Msg("qdrant query failed")
		}
		retrievedCodeCount = len(chunks)
		req.ReportProgress("qdrant_query_complete", "Qdrant query complete", 25)

		if p.rerankEnabled {
			chunks = rerankChunks(ctx, p.reranker, req.QueryText, chunks, req.Limit, p.log)
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
	messages := buildPrompt(req, changeChunks, codeChunks)
	maxTokens := ResolveTokenBudget(req, messages)
	req.ReportProgress("messages_assembled", "Assembled vLLM messages", 50)

	// 4. Call LLM
	p.log.Info().
		Str("messages_sha256", hashMessages(messages)).
		Int("message_count", len(messages)).
		Int("max_tokens", maxTokens).
		Interface("code_evidence", evidenceCounts).
		Msg("assembled vllm messages")
	answer, err := p.vllm.Complete(ctx, messages, maxTokens)
	if err != nil {
		return nil, fmt.Errorf("vllm complete: %w", err)
	}
	req.ReportProgress("vllm_complete", "vLLM completion complete", 80)

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
			req.ReportProgress("vllm_complete", "vLLM format repair complete", 80)
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
	vector, err := p.embedder.Embed(ctx, req.QueryText)
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
		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vector, req.RepoID, req.Component, queryLimit)
		codeCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.docCollection, vector, req.RepoID, req.Component, queryLimit)
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
