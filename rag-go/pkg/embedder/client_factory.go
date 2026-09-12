package embedder

import (
	"context"
	"strings"
	"time"

	"github.com/rs/zerolog"
)

// EmbeddingClient is the common interface used by the RAG pipeline.
type EmbeddingClient interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// NewClientFromType picks the embedding client implementation by type.
func NewClientFromType(clientType string, baseURL string, timeout time.Duration, log zerolog.Logger) EmbeddingClient {
	switch strings.ToLower(strings.TrimSpace(clientType)) {
	case "gte-qwen":
		return NewGTEQwenClient(baseURL, timeout, log)
	case "e5":
		return NewClient(baseURL, timeout, log)
	default:
		log.Fatal().Str("embed_client_type", clientType).Msg("unknown embed client type")
		return nil
	}
}
