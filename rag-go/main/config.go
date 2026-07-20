package main

import (
	"fmt"
	"os"
	"strings"
)

type config struct {
	FiberPort        string
	QdrantHost       string // host or host:port
	VLLMHost         string // host or host:port
	VLLMTransport    string // "grpc" or "http"
	EmbedHost        string // host or host:port
	ModelName        string
	ChangeCollection string
	CodeCollection   string
}

func loadConfig() config {
	return config{
		FiberPort:        getEnv("FIBER_PORT", ":8080"),
		QdrantHost:       normalizeHostPort(getEnv("QDRANT_HOST", "qdrant-service"), 6334),
		VLLMHost:         normalizeHostPort(getEnv("VLLM_HOST", "qwen-3-service"), 50051),
		VLLMTransport:    getEnv("VLLM_TRANSPORT", "grpc"),
		EmbedHost:        normalizeHostPort(getEnv("EMBED_SERVICE_HOST", "embed-e5-service"), 8000),
		ModelName:        getEnv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
		ChangeCollection: getEnv("CHANGE_COLLECTION", "change_chunks"),
		CodeCollection:   getEnv("CODE_COLLECTION", "code_chunks"),
	}
}

// normalizeHostPort ensures host:port format, using defaultPort if no port specified.
func normalizeHostPort(hostPort string, defaultPort int) string {
	hostPort = strings.TrimSpace(hostPort)

	// Already has port
	if strings.Contains(hostPort, ":") {
		return hostPort
	}

	// No port, append default
	return fmt.Sprintf("%s:%d", hostPort, defaultPort)
}

// buildHTTPURL constructs http://host:port from config.
func buildHTTPURL(hostPort string) string {
	if strings.HasPrefix(hostPort, "http://") || strings.HasPrefix(hostPort, "https://") {
		return hostPort
	}
	return "http://" + hostPort
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
