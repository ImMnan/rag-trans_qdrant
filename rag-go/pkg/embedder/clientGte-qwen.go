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

func NewGTEQwenClient(baseURL string, timeout time.Duration, log zerolog.Logger) *GTEQwenClient {
	return &GTEQwenClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
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

// Rerank scores each document's relevance to query using the embed service's cross-encoder.
func (c *GTEQwenClient) Rerank(ctx context.Context, query string, documents []string) ([]float32, error) {
	body, err := json.Marshal(rerankRequest{Query: query, Documents: documents})
	if err != nil {
		return nil, fmt.Errorf("marshal rerank request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create rerank request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rerank request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("rerank service returned %d", resp.StatusCode)
	}

	var result rerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode rerank response: %w", err)
	}

	c.log.Debug().Str("client_type", "gte-qwen").Int("scored", len(result.Scores)).Msg("rerank scores received")
	return result.Scores, nil
}

type sparseRequest struct {
	Text string `json:"text"`
}

type sparseResponse struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// SparseEmbed encodes query text into the BM25 sparse vector used for hybrid retrieval.
func (c *GTEQwenClient) SparseEmbed(ctx context.Context, text string) ([]uint32, []float32, error) {
	body, err := json.Marshal(sparseRequest{Text: text})
	if err != nil {
		return nil, nil, fmt.Errorf("marshal sparse request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/sparse", bytes.NewReader(body))
	if err != nil {
		return nil, nil, fmt.Errorf("create sparse request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("sparse request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("sparse service returned %d", resp.StatusCode)
	}

	var result sparseResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, nil, fmt.Errorf("decode sparse response: %w", err)
	}
	if len(result.Indices) != len(result.Values) {
		return nil, nil, fmt.Errorf("sparse response has %d indices but %d values", len(result.Indices), len(result.Values))
	}

	c.log.Debug().Str("client_type", "gte-qwen").Int("sparse_terms", len(result.Indices)).Msg("sparse embedding received")
	return result.Indices, result.Values, nil
}
