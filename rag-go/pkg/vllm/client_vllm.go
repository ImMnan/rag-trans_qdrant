package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
	"github.com/rs/zerolog"
)

// ── Shared HTTP types (used by HTTPClient) ────────────────────────────────────

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float32       `json:"temperature"`
	TopP        float32       `json:"top_p"`
	Seed        int           `json:"seed"`
	TokenLimit  int           `json:"max_tokens,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
}

// ── HTTP client ───────────────────────────────────────────────────────────────

// HTTPClient calls vLLM via the OpenAI-compatible HTTP API (/v1/chat/completions).
type HTTPClient struct {
	baseURL    string
	modelName  string
	httpClient *http.Client
	log        zerolog.Logger
}

const fixedSeed = 42

func NewHTTPClient(baseURL string, modelName string, timeout time.Duration, log zerolog.Logger) *HTTPClient {
	return &HTTPClient{
		baseURL:   baseURL,
		modelName: modelName,
		httpClient: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     timeout * 3 / 4, // 75% of client timeout
			},
		},
		log: log,
	}
}

// Complete sends messages to vLLM via the OpenAI-compatible HTTP API.
func (c *HTTPClient) Complete(ctx context.Context, messages []pipeline.Message, maxTokens int) (string, error) {
	chatMsgs := make([]chatMessage, len(messages))
	for i, m := range messages {
		chatMsgs[i] = chatMessage{Role: m.Role, Content: m.Content}
	}

	body, err := json.Marshal(chatRequest{
		Model:       c.modelName,
		Messages:    chatMsgs,
		Temperature: 0,
		TopP:        1,
		Seed:        fixedSeed,
		TokenLimit:  maxTokens,
	})
	if err != nil {
		return "", fmt.Errorf("marshal vllm request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create vllm request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("vllm request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		c.log.Error().Int("status", resp.StatusCode).Str("body", string(errBody)).Int("max_tokens", maxTokens).Msg("vllm returned error")
		return "", fmt.Errorf("vllm returned %d: %s", resp.StatusCode, string(errBody))
	}

	var result chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode vllm response: %w", err)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("vllm returned empty choices")
	}

	answer := result.Choices[0].Message.Content
	c.log.Debug().Int("response_len", len(answer)).Msg("vllm http completion received")
	return answer, nil
}
