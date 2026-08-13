package embedder

import (
	"context"
	"strings"

	"github.com/rs/zerolog"
)

// EmbeddingClient is the common interface used by the RAG pipeline.
type EmbeddingClient interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// NewClientFromType picks the embedding client implementation by type.
func NewClientFromType(clientType string, baseURL string, log zerolog.Logger) EmbeddingClient {
	switch strings.ToLower(strings.TrimSpace(clientType)) {
	case "gte-qwen":
		return NewGTEQwenClient(baseURL, log)
	case "e5", "":
		return NewClient(baseURL, log)
	default:
		log.Warn().Str("embed_client_type", clientType).Msg("unknown embed client type, falling back to e5")
		return NewClient(baseURL, log)
	}
}
