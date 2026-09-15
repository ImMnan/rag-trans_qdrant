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
	qdrant                   QdrantQuerier
	vllm                     VLLMCompleter
	embedder                 Embedder
	docProcessor             DocProcessor
	changeCollection         string
	codeCollection           string
	docCollection            string
	genDocCollection         string
	contextTruncationEnabled bool
	appProfileDir            string
	appProfileFiles          map[string]string
	log                      zerolog.Logger
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
	contextTruncationEnabled bool,
	changeCollection string,
	codeCollection string,
	docCollection string,
	genDocCollection string,
	appProfileDir string,
	appProfileFiles map[string]string,
) *DOCPipeline {
	docProcessor := NewLLMDocProcessor(vllm, NewDefaultDocDecisionEngine())

	return &DOCPipeline{
		qdrant:                   qdrant,
		vllm:                     vllm,
		embedder:                 embedder,
		docProcessor:             docProcessor,
		changeCollection:         changeCollection,
		codeCollection:           codeCollection,
		docCollection:            docCollection,
		genDocCollection:         genDocCollection,
		contextTruncationEnabled: contextTruncationEnabled,
		appProfileDir:            appProfileDir,
		appProfileFiles:          appProfileFiles,
		log:                      zerolog.Nop(),
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

// Execute runs the full RAG pipeline for a single request.
func (p *RAGPipeline) Execute(ctx context.Context, req Request) (*Response, error) {
	isStandard := strings.EqualFold(strings.TrimSpace(req.Type), "standard")
	if !isStandard {
		req.AppProfile = resolveAppProfile(p.appProfileDir, p.appProfileFiles, req.RepoID, p.log)
	}

	// 1. Embed the query once
	vector, err := p.embedder.Embed(ctx, req.QueryText)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	// 2. Fan-out: query both the collections concurrently. When reranking is enabled,
	// over-fetch a larger candidate pool so the cross-encoder has more to choose from.
	queryLimit := req.Limit
	if p.rerankEnabled && p.rerankOverfetchMultiplier > 1 {
		queryLimit = req.Limit * p.rerankOverfetchMultiplier
	}

	type result struct {
		chunks []string
		err    error
	}

	var wg sync.WaitGroup
	changeCh := make(chan result, 1)
	codeCh := make(chan result, 1)
	wg.Add(2)
	go func() {
		defer wg.Done()
		var chunks []string
		var err error
		if isStandard {
			chunks, err = p.qdrant.QueryStandardAll(ctx, p.changeCollection, req.RepoID, req.Component, req.FromDate, req.ToDate, p.changeDateField)
		} else {
			chunks, err = p.qdrant.Query(ctx, p.changeCollection, vector, req.RepoID, req.Component, queryLimit)
		}
		changeCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vector, req.RepoID, req.Component, queryLimit)
		codeCh <- result{chunks, err}
	}()
	wg.Wait()

	changeResult := <-changeCh
	codeResult := <-codeCh

	if changeResult.err != nil {
		p.log.Warn().Err(changeResult.err).Str("collection", p.changeCollection).Msg("qdrant query failed")
	}
	if codeResult.err != nil {
		p.log.Warn().Err(codeResult.err).Str("collection", p.codeCollection).Msg("qdrant query failed")
	}

	retrievedChangeCount := len(changeResult.chunks)
	retrievedCodeCount := len(codeResult.chunks)

	// 3. Rerank each side's candidate pool down to req.Limit with the cross-encoder.
	if p.rerankEnabled {
		if !isStandard {
			changeResult.chunks = rerankChunks(ctx, p.reranker, req.QueryText, changeResult.chunks, req.Limit, p.log)
		}
		codeResult.chunks = rerankChunks(ctx, p.reranker, req.QueryText, codeResult.chunks, req.Limit, p.log)
	}

	// 4. Build prompt
	changeChunks := changeResult.chunks
	codeBudget := maxContextCharsTotal - chunksCharSize(changeChunks)
	codeChunks := codeResult.chunks
	if p.contextTruncationEnabled {
		codeChunks = TruncateChunksToCharBudget(codeResult.chunks, codeBudget)
	}
	contextTruncationActive := len(codeChunks) < len(codeResult.chunks)
	if contextTruncationActive {
		p.log.Warn().
			Int("change_chunks_kept", len(changeChunks)).
			Int("change_chunks_retrieved", len(changeResult.chunks)).
			Int("code_chunks_kept", len(codeChunks)).
			Int("code_chunks_retrieved", len(codeResult.chunks)).
			Msg("truncated retrieved chunks to stay within context budget")
	}
	codeChunks, evidenceCounts := annotateCodeChunks(codeChunks)
	messages := buildPrompt(req, changeChunks, codeChunks)
	maxTokens := ResolveTokenBudget(req, messages)

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

	// 2. Fan-out: query both the collections concurrently
	type result struct {
		chunks []string
		err    error
	}

	var wg sync.WaitGroup
	changeCh := make(chan result, 1)
	codeCh := make(chan result, 1)
	docCh := make(chan result, 1)
	genDocCh := make(chan result, 1)
	wg.Add(4)
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.changeCollection, vector, req.RepoID, req.Component, req.Limit)
		changeCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vector, req.RepoID, req.Component, req.Limit)
		codeCh <- result{chunks, err}
	}()

	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.docCollection, vector, req.RepoID, req.Component, req.Limit)
		docCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.genDocCollection, vector, req.RepoID, req.Component, req.Limit)
		genDocCh <- result{chunks, err}
	}()
	wg.Wait()

	changeResult := <-changeCh
	codeResult := <-codeCh
	docResult := <-docCh
	genDocResult := <-genDocCh

	if changeResult.err != nil {
		p.log.Warn().Err(changeResult.err).Str("collection", p.changeCollection).Msg("qdrant query failed")
	}
	if codeResult.err != nil {
		p.log.Warn().Err(codeResult.err).Str("collection", p.codeCollection).Msg("qdrant query failed")
	}
	if docResult.err != nil {
		p.log.Warn().Err(docResult.err).Str("collection", p.docCollection).Msg("qdrant query failed")
	}
	if genDocResult.err != nil {
		p.log.Warn().Err(genDocResult.err).Str("collection", p.genDocCollection).Msg("qdrant query failed")
	}

	// 3. Cap non-change sides against the whole pool as a backstop only. The doc workflow
	// splits its context across two calls, and compose fitting preserves change evidence
	// while shedding documentation and code context around it.
	changeChunks := changeResult.chunks
	codeChunks := codeResult.chunks
	docChunks := docResult.chunks
	genDocChunks := genDocResult.chunks
	if p.contextTruncationEnabled {
		codeChunks = TruncateChunksToCharBudget(codeResult.chunks, maxContextCharsTotal-chunksCharSize(changeChunks))
		docChunks = TruncateChunksToCharBudget(docResult.chunks, maxContextCharsTotal)
		genDocChunks = TruncateChunksToCharBudget(genDocResult.chunks, maxContextCharsTotal)
	}
	contextTruncationActive := len(codeChunks) < len(codeResult.chunks) ||
		len(docChunks) < len(docResult.chunks) || len(genDocChunks) < len(genDocResult.chunks)
	if contextTruncationActive {
		p.log.Warn().
			Int("change_chunks_kept", len(changeChunks)).
			Int("change_chunks_retrieved", len(changeResult.chunks)).
			Int("code_chunks_kept", len(codeChunks)).
			Int("code_chunks_retrieved", len(codeResult.chunks)).
			Int("doc_chunks_kept", len(docChunks)).
			Int("doc_chunks_retrieved", len(docResult.chunks)).
			Int("gen_doc_chunks_kept", len(genDocChunks)).
			Int("gen_doc_chunks_retrieved", len(genDocResult.chunks)).
			Msg("truncated retrieved chunks to stay within context budget")
	}

	// 4. Run strong-confidence doc workflow (extract -> audit -> decide -> generate -> validate)
	answer, err := p.docProcessor.Process(ctx, req, changeChunks, codeChunks, docChunks, genDocChunks)
	if err != nil {
		return nil, fmt.Errorf("doc workflow: %w", err)
	}

	return &Response{
		Answer: answer,
		Sources: map[string]int{
			"change_chunks_retrieved":  len(changeResult.chunks),
			"code_chunks_retrieved":    len(codeResult.chunks),
			"doc_chunks_retrieved":     len(docResult.chunks),
			"gen_doc_chunks_retrieved": len(genDocResult.chunks),
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
