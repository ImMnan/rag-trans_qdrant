package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	vllmpb "github.com/immnan/rag-trans_qdrant/rag-go/gen/vllm"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
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

// ── gRPC client ───────────────────────────────────────────────────────────────

// GRPCClient calls vLLM via its native gRPC Generate service.
type GRPCClient struct {
	grpc      vllmpb.GenerateClient
	modelName string
	log       zerolog.Logger
}

func NewGRPCClient(host string, modelName string, log zerolog.Logger) *GRPCClient {
	conn, err := grpc.NewClient(host, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error().Err(err).Str("host", host).Msg("vllm gRPC connection failed")
	}
	return &GRPCClient{
		grpc:      vllmpb.NewGenerateClient(conn),
		modelName: modelName,
		log:       log,
	}
}

// Complete applies the Qwen2.5 chat template and calls vLLM via gRPC.
func (c *GRPCClient) Complete(ctx context.Context, messages []pipeline.Message) (string, error) {
	temperature := inferTemperature(messages)

	req := &vllmpb.GenerateRequest{
		Model:       c.modelName,
		Prompt:      &vllmpb.GenerateRequest_Text{Text: formatQwenPrompt(messages)},
		Temperature: &temperature,
		Stopping:    &vllmpb.StoppingCriteria{MaxNewTokens: 2048},
		Response:    &vllmpb.ResponseOptions{OutputText: boolPtr(true)},
	}

	resp, err := c.grpc.Generate(ctx, req)
	if err != nil {
		return "", fmt.Errorf("vllm grpc generate: %w", err)
	}
	if resp.Outputs == nil {
		return "", fmt.Errorf("vllm returned nil output")
	}

	answer := resp.Outputs.Text
	c.log.Debug().Int("response_len", len(answer)).Uint32("tokens", resp.Outputs.NumTokens).Msg("vllm grpc completion received")
	return answer, nil
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

// formatQwenPrompt applies the Qwen2.5 chat template to a message slice.
// Used by the gRPC client since vLLM's gRPC API takes raw text, not chat messages.
func formatQwenPrompt(messages []pipeline.Message) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(m.Role)
		sb.WriteString("\n")
		sb.WriteString(m.Content)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

func boolPtr(b bool) *bool { return &b }
