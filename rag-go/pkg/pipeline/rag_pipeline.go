package pipeline

import (
	"context"
	"fmt"
	"sync"

	"github.com/rs/zerolog"
)

// Interfaces — swap real clients for mocks in tests.
type QdrantQuerier interface {
	Query(ctx context.Context, collection string, vector []float32, repoID string, limit int) ([]string, error)
}

type VLLMCompleter interface {
	Complete(ctx context.Context, messages []Message) (string, error)
}

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// Message is a minimal chat message passed to the LLM.
type Message struct {
	Role    string
	Content string
}

// Request is the pipeline's input — decoupled from the HTTP layer.
type Request struct {
	QueryText string
	RepoID    string
	Type      string
	Limit     int
}

// Response is what the pipeline returns to the handler.
type Response struct {
	Answer  string         `json:"answer"`
	Sources map[string]int `json:"sources"`
}

// RAGPipeline wires all downstream clients together.
type RAGPipeline struct {
	qdrant           QdrantQuerier
	vllm             VLLMCompleter
	embedder         Embedder
	changeCollection string
	codeCollection   string
	log              zerolog.Logger
}

// RAGPipeline wires all downstream clients together.
type DOCPipeline struct {
	qdrant           QdrantQuerier
	vllm             VLLMCompleter
	embedder         Embedder
	docProcessor     DocProcessor
	changeCollection string
	codeCollection   string
	docCollection    string
	genDocCollection string
	log              zerolog.Logger
}

func New(
	qdrant QdrantQuerier,
	vllm VLLMCompleter,
	embedder Embedder,
	changeCollection string,
	codeCollection string,
) *RAGPipeline {
	return &RAGPipeline{
		qdrant:           qdrant,
		vllm:             vllm,
		embedder:         embedder,
		changeCollection: changeCollection,
		codeCollection:   codeCollection,
		log:              zerolog.Nop(),
	}
}

func NewDoc(
	qdrant QdrantQuerier,
	vllm VLLMCompleter,
	embedder Embedder,
	changeCollection string,
	codeCollection string,
	docCollection string,
	genDocCollection string,
) *DOCPipeline {
	docProcessor := NewLLMDocProcessor(vllm, NewDefaultDocDecisionEngine())

	return &DOCPipeline{
		qdrant:           qdrant,
		vllm:             vllm,
		embedder:         embedder,
		docProcessor:     docProcessor,
		changeCollection: changeCollection,
		codeCollection:   codeCollection,
		docCollection:    docCollection,
		genDocCollection: genDocCollection,
		log:              zerolog.Nop(),
	}
}

func (p *RAGPipeline) WithLogger(log zerolog.Logger) *RAGPipeline {
	p.log = log
	return p
}

type Execution interface {
	Execute(ctx context.Context, req Request) (*Response, error)
}

// Execute runs the full RAG pipeline for a single request.
func (p *RAGPipeline) Execute(ctx context.Context, req Request) (*Response, error) {
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
	wg.Add(2)
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.changeCollection, vector, req.RepoID, req.Limit)
		changeCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vector, req.RepoID, req.Limit)
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

	// 3. Build prompt
	messages := buildPrompt(req, changeResult.chunks, codeResult.chunks)

	// 4. Call LLM
	answer, err := p.vllm.Complete(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("vllm complete: %w", err)
	}

	return &Response{
		Answer: answer,
		Sources: map[string]int{
			"change_chunks_retrieved": len(changeResult.chunks),
			"code_chunks_retrieved":   len(codeResult.chunks),
		},
	}, nil
}

func (p *DOCPipeline) Execute(ctx context.Context, req Request) (*Response, error) {
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
		chunks, err := p.qdrant.Query(ctx, p.changeCollection, vector, req.RepoID, req.Limit)
		changeCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.codeCollection, vector, req.RepoID, req.Limit)
		codeCh <- result{chunks, err}
	}()

	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.docCollection, vector, req.RepoID, req.Limit)
		docCh <- result{chunks, err}
	}()
	go func() {
		defer wg.Done()
		chunks, err := p.qdrant.Query(ctx, p.genDocCollection, vector, req.RepoID, req.Limit)
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

	// 3. Run strong-confidence doc workflow (extract -> audit -> decide -> generate -> validate)
	answer, err := p.docProcessor.Process(ctx, req, changeResult.chunks, codeResult.chunks, docResult.chunks, genDocResult.chunks)
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
	}, nil
}
