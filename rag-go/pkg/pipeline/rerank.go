package pipeline

import (
	"context"
	"sort"

	"github.com/rs/zerolog"
)

type scoredChunk struct {
	chunk string
	score float32
}

// rerankChunks scores chunks against query with a cross-encoder and returns the top `keep`
// chunks in descending relevance order. On any failure (or if reranker is nil), it falls
// back to the head of chunks in retrieval order so a reranker outage never breaks retrieval.
func rerankChunks(ctx context.Context, reranker Reranker, query string, chunks []string, keep int, log zerolog.Logger) []string {
	if reranker == nil || len(chunks) == 0 {
		return truncateHead(chunks, keep)
	}

	scores, err := reranker.Rerank(ctx, query, chunks)
	if err != nil || len(scores) != len(chunks) {
		log.Warn().Err(err).Int("chunks", len(chunks)).Msg("rerank failed, falling back to original retrieval order")
		return truncateHead(chunks, keep)
	}

	ranked := make([]scoredChunk, len(chunks))
	for i, c := range chunks {
		ranked[i] = scoredChunk{chunk: c, score: scores[i]}
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })

	if keep > len(ranked) {
		keep = len(ranked)
	}
	if keep < 0 {
		keep = 0
	}
	out := make([]string, keep)
	for i := 0; i < keep; i++ {
		out[i] = ranked[i].chunk
	}
	return out
}

func truncateHead(chunks []string, keep int) []string {
	if keep < 0 {
		keep = 0
	}
	if keep >= len(chunks) {
		return chunks
	}
	return chunks[:keep]
}
