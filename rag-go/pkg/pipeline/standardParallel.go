package pipeline

import (
	"context"
	"fmt"
	"sync"
)

// maxParallelLLMCalls caps concurrent vLLM requests per standard query so one large
// window cannot monopolize the model server.
const maxParallelLLMCalls = 4

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
