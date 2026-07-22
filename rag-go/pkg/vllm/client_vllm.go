package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func NewHTTPClient(baseURL string, modelName string, log zerolog.Logger) *HTTPClient {
	return &HTTPClient{
		baseURL:   baseURL,
		modelName: modelName,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		log: log,
	}
}

// Complete sends messages to vLLM via the OpenAI-compatible HTTP API.
func (c *HTTPClient) Complete(ctx context.Context, messages []pipeline.Message) (string, error) {
	temperature := inferTemperature(messages)

	chatMsgs := make([]chatMessage, len(messages))
	for i, m := range messages {
		chatMsgs[i] = chatMessage{Role: m.Role, Content: m.Content}
	}

	body, err := json.Marshal(chatRequest{Model: c.modelName, Messages: chatMsgs, Temperature: temperature})
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
		return "", fmt.Errorf("vllm returned %d", resp.StatusCode)
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

// ── Shared helpers ────────────────────────────────────────────────────────────

// inferTemperature returns 0.3 for standard (system prompt) requests, 0.1 for direct.
func inferTemperature(messages []pipeline.Message) float32 {
	if len(messages) > 1 && messages[0].Role == "system" {
		return 0.3
	}
	return 0.1
}
