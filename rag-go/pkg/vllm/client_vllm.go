package vllm

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	smgcommon "github.com/immnan/rag-trans_qdrant/rag-go/gen/smgcommon"
	vllmengine "github.com/immnan/rag-trans_qdrant/rag-go/gen/vllmengine"
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
	grpc      vllmengine.VllmEngineClient
	modelName string
	log       zerolog.Logger

	tokenizerMu sync.RWMutex
	tokenizer   *tokenizer.Tokenizer
}

func NewGRPCClient(host string, modelName string, log zerolog.Logger) *GRPCClient {
	conn, err := grpc.NewClient(host, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Error().Err(err).Str("host", host).Msg("vllm gRPC connection failed")
	}
	return &GRPCClient{
		grpc:      vllmengine.NewVllmEngineClient(conn),
		modelName: modelName,
		log:       log,
	}
}

// Complete applies the chat template and calls vLLM Engine gRPC Generate stream.
func (c *GRPCClient) Complete(ctx context.Context, messages []pipeline.Message) (string, error) {
	temperature := inferTemperature(messages)
	maxTokens := uint32(2048)

	req := &vllmengine.GenerateRequest{
		RequestId: fmt.Sprintf("rag-go-%d", time.Now().UnixNano()),
		Input:     &vllmengine.GenerateRequest_Text{Text: formatQwenPrompt(messages)},
		Stream:    false,
		SamplingParams: &vllmengine.SamplingParams{
			Temperature: &temperature,
			MaxTokens:   &maxTokens,
			N:           1,
		},
	}

	stream, err := c.grpc.Generate(ctx, req)
	if err != nil {
		return "", fmt.Errorf("vllm grpc generate: %w", err)
	}

	var completion *vllmengine.GenerateComplete
	for {
		resp, recvErr := stream.Recv()
		if recvErr == io.EOF {
			break
		}
		if recvErr != nil {
			return "", fmt.Errorf("vllm grpc stream recv: %w", recvErr)
		}

		if complete := resp.GetComplete(); complete != nil {
			completion = complete
		}
	}

	if completion == nil {
		return "", fmt.Errorf("vllm returned no completion payload")
	}

	answer := c.decodeOutput(ctx, completion.OutputIds)
	c.log.Debug().
		Int("response_len", len(answer)).
		Int("tokens", len(completion.OutputIds)).
		Str("finish_reason", completion.FinishReason).
		Msg("vllm grpc completion received")
	return answer, nil
}

func (c *GRPCClient) decodeOutput(ctx context.Context, ids []uint32) string {
	if len(ids) == 0 {
		return ""
	}

	tk, err := c.getOrInitTokenizer(ctx)
	if err != nil {
		c.log.Warn().Err(err).Msg("vllm tokenizer unavailable; falling back to token ID output")
		return tokenIDsToText(ids)
	}

	intIDs := make([]int, len(ids))
	for i, id := range ids {
		intIDs[i] = int(id)
	}

	decoded := strings.TrimSpace(tk.Decode(intIDs, true))
	if decoded == "" {
		c.log.Warn().Msg("vllm tokenizer produced empty decode; falling back to token IDs")
		return tokenIDsToText(ids)
	}

	return decoded
}

func (c *GRPCClient) getOrInitTokenizer(ctx context.Context) (*tokenizer.Tokenizer, error) {
	c.tokenizerMu.RLock()
	if c.tokenizer != nil {
		tk := c.tokenizer
		c.tokenizerMu.RUnlock()
		return tk, nil
	}
	c.tokenizerMu.RUnlock()

	c.tokenizerMu.Lock()
	defer c.tokenizerMu.Unlock()
	if c.tokenizer != nil {
		return c.tokenizer, nil
	}

	stream, err := c.grpc.GetTokenizer(ctx, &smgcommon.GetTokenizerRequest{})
	if err != nil {
		return nil, fmt.Errorf("get tokenizer stream: %w", err)
	}

	var zipData bytes.Buffer
	var serverSHA string

	for {
		chunk, recvErr := stream.Recv()
		if recvErr == io.EOF {
			break
		}
		if recvErr != nil {
			return nil, fmt.Errorf("receive tokenizer chunk: %w", recvErr)
		}

		if _, writeErr := zipData.Write(chunk.GetData()); writeErr != nil {
			return nil, fmt.Errorf("buffer tokenizer chunk: %w", writeErr)
		}

		if sha := strings.TrimSpace(chunk.GetSha256()); sha != "" {
			serverSHA = sha
		}
	}

	if zipData.Len() == 0 {
		return nil, fmt.Errorf("tokenizer payload was empty")
	}

	if serverSHA != "" {
		sum := sha256.Sum256(zipData.Bytes())
		localSHA := hex.EncodeToString(sum[:])
		if !strings.EqualFold(localSHA, serverSHA) {
			return nil, fmt.Errorf("tokenizer sha256 mismatch: got %s want %s", localSHA, serverSHA)
		}
	}

	workDir, err := os.MkdirTemp("", "rag-go-vllm-tokenizer-")
	if err != nil {
		return nil, fmt.Errorf("create tokenizer temp dir: %w", err)
	}

	tokenizerPath, err := unzipTokenizerArtifact(zipData.Bytes(), workDir)
	if err != nil {
		return nil, err
	}

	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer from %s: %w", tokenizerPath, err)
	}

	c.tokenizer = tk
	c.log.Info().Str("tokenizer_path", tokenizerPath).Msg("initialized vllm tokenizer for gRPC decode")
	return tk, nil
}

func unzipTokenizerArtifact(zipBytes []byte, destDir string) (string, error) {
	reader, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	if err != nil {
		return "", fmt.Errorf("open tokenizer zip: %w", err)
	}

	var tokenizerPath string
	for _, f := range reader.File {
		cleanName := filepath.Clean(f.Name)
		if cleanName == "." || strings.HasPrefix(cleanName, "..") {
			continue
		}

		targetPath := filepath.Join(destDir, cleanName)
		if !strings.HasPrefix(targetPath, destDir+string(os.PathSeparator)) && targetPath != destDir {
			continue
		}

		if f.FileInfo().IsDir() {
			if mkErr := os.MkdirAll(targetPath, 0o755); mkErr != nil {
				return "", fmt.Errorf("create tokenizer dir: %w", mkErr)
			}
			continue
		}

		if mkErr := os.MkdirAll(filepath.Dir(targetPath), 0o755); mkErr != nil {
			return "", fmt.Errorf("create tokenizer parent dir: %w", mkErr)
		}

		src, openErr := f.Open()
		if openErr != nil {
			return "", fmt.Errorf("open tokenizer zip entry: %w", openErr)
		}

		dst, createErr := os.Create(targetPath)
		if createErr != nil {
			src.Close()
			return "", fmt.Errorf("create tokenizer file: %w", createErr)
		}

		_, copyErr := io.Copy(dst, src)
		closeDstErr := dst.Close()
		closeSrcErr := src.Close()
		if copyErr != nil {
			return "", fmt.Errorf("write tokenizer file: %w", copyErr)
		}
		if closeDstErr != nil {
			return "", fmt.Errorf("finalize tokenizer file: %w", closeDstErr)
		}
		if closeSrcErr != nil {
			return "", fmt.Errorf("close tokenizer zip entry: %w", closeSrcErr)
		}

		if filepath.Base(targetPath) == "tokenizer.json" && tokenizerPath == "" {
			tokenizerPath = targetPath
		}
	}

	if tokenizerPath == "" {
		return "", fmt.Errorf("tokenizer.json not found in tokenizer artifact")
	}

	return tokenizerPath, nil
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

func tokenIDsToText(ids []uint32) string {
	if len(ids) == 0 {
		return ""
	}

	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.FormatUint(uint64(id), 10)
	}

	return "TOKENS[" + strings.Join(parts, " ") + "]"
}
