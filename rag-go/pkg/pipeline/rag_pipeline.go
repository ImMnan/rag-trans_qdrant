package pipeline

import (
	"context"
	"fmt"
	"strings"
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

func (p *RAGPipeline) WithLogger(log zerolog.Logger) *RAGPipeline {
	p.log = log
	return p
}

// Execute runs the full RAG pipeline for a single request.
func (p *RAGPipeline) Execute(ctx context.Context, req Request) (*Response, error) {
	// 1. Embed the query once
	vector, err := p.embedder.Embed(ctx, req.QueryText)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	// 2. Fan-out: query both collections concurrently
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

// buildPrompt assembles the LLM messages from retrieved chunks.
// Mirrors the two-mode prompt strategy from the original Python pipeline.
func buildPrompt(req Request, changeChunks, codeChunks []string) []Message {
	changeCtx := joinChunks(changeChunks, "No change data found.")
	codeCtx := joinChunks(codeChunks, "No source context found.")

	if strings.EqualFold(strings.TrimSpace(req.Type), "standard") {
		systemPrompt := "You are a senior engineer producing product release summaries. " +
			"Use only the provided context. Structure your answer with these sections:\n" +
			"1. **What Changed** - describe the commits/diffs concisely.\n" +
			"2. **User Impact** - explain what end-users will notice or need to act on.\n" +
			"3. **Security & Performance** - flag any security fixes or performance optimizations; " +
			"write 'None identified' if absent."

		userPrompt := fmt.Sprintf("## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Request\n%s",
			changeCtx, codeCtx, req.QueryText)

		return []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		}
	}

	directPrompt := fmt.Sprintf(
		"Answer the user question using only the context below. "+
			"Be concise and factual. If asked whether a feature is supported, answer with 'Yes' or 'No' "+
			"and include when it first appears in the provided context if available; "+
			"otherwise say 'Unknown based on provided context'.\n\n"+
			"## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Question\n%s",
		changeCtx, codeCtx, req.QueryText,
	)

	return []Message{{Role: "user", Content: directPrompt}}
}

func joinChunks(chunks []string, fallback string) string {
	if len(chunks) == 0 {
		return fallback
	}
	return strings.Join(chunks, "\n---\n")
}
