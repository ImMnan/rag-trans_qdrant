package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/rs/zerolog"
)

// GTEQwenClient calls the embedding service and explicitly marks query input type.
type GTEQwenClient struct {
	baseURL    string
	httpClient *http.Client
	log        zerolog.Logger
}

func NewGTEQwenClient(baseURL string, log zerolog.Logger) *GTEQwenClient {
	return &GTEQwenClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        50,
				MaxIdleConnsPerHost: 50,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		log: log,
	}
}

type gteQwenEmbedRequest struct {
	Text      string `json:"text"`
	InputType string `json:"input_type"`
}

type gteQwenEmbedResponse struct {
	Vector []float32 `json:"vector"`
}

// Embed sends a retrieval query text for embedding.
func (c *GTEQwenClient) Embed(ctx context.Context, text string) ([]float32, error) {
	body, err := json.Marshal(gteQwenEmbedRequest{Text: text, InputType: "query"})
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed service returned %d", resp.StatusCode)
	}

	var result gteQwenEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	c.log.Debug().Str("client_type", "gte-qwen").Int("vector_len", len(result.Vector)).Msg("embedding received")
	return result.Vector, nil
}
