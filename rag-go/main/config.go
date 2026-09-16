package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type config struct {
	FiberPort                 string
	ReadTimeout               time.Duration
	WriteTimeout              time.Duration
	IdleTimeout               time.Duration
	VLLMTimeout               time.Duration
	EmbedTimeout              time.Duration
	QdrantHost                string // host or host:port
	QdrantScoreThreshold      float32
	QdrantNeighborStitch      bool
	QdrantMMREnabled          bool
	QdrantMMRLambda           float32
	QdrantMMROverfetch        int
	RerankEnabled             bool
	RerankOverfetchMultiplier int
	ContextTruncationEnabled  bool
	VLLMHost                  string // host or host:port
	EmbedClientType           string
	EmbedHost                 string // host or host:port
	ModelName                 string
	ChangeCollection          string
	ChangeDateField           string
	CodeCollection            string
	DocCollection             string
	AppProfileDir             string
	AppProfileFiles           map[string]string
}

func loadConfig() (config, error) {
	embedClientType, embedHost, err := parseEmbedServiceType(getEnv("EMBED_SERVICE_TYPE", ""))
	if err != nil {
		return config{}, err
	}

	return config{
		FiberPort:                 getEnv("FIBER_PORT", "8080"),
		ReadTimeout:               getEnvDuration("FIBER_READ_TIMEOUT", 30*time.Second),
		WriteTimeout:              getEnvDuration("FIBER_WRITE_TIMEOUT", 120*time.Second),
		IdleTimeout:               getEnvDuration("FIBER_IDLE_TIMEOUT", 60*time.Second),
		VLLMTimeout:               getEnvDuration("VLLM_TIMEOUT", 120*time.Second),
		EmbedTimeout:              getEnvDuration("EMBED_TIMEOUT", 60*time.Second),
		QdrantHost:                normalizeHostPort(getEnv("QDRANT_HOST", "qdrant-service"), 6334),
		QdrantScoreThreshold:      getEnvFloat32("QDRANT_SCORE_THRESHOLD", 0),
		QdrantNeighborStitch:      getEnvBool("QDRANT_NEIGHBOR_STITCH", true),
		QdrantMMREnabled:          getEnvBool("QDRANT_MMR_ENABLED", true),
		QdrantMMRLambda:           getEnvFloat32("QDRANT_MMR_LAMBDA", 0.7),
		QdrantMMROverfetch:        getEnvInt("QDRANT_MMR_OVERFETCH", 3),
		RerankEnabled:             getEnvBool("RERANK_ENABLED", false),
		RerankOverfetchMultiplier: getEnvInt("RERANK_OVERFETCH_MULTIPLIER", 4),
		ContextTruncationEnabled:  getEnvBool("CONTEXT_TRUNCATION_ENABLED", true),
		VLLMHost:                  normalizeHostPort(getEnv("VLLM_HOST", "qwen-3-service"), 80),
		EmbedClientType:           embedClientType,
		EmbedHost:                 normalizeHostPort(embedHost, 80),
		ModelName:                 getEnv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
		ChangeCollection:          getEnv("CHANGE_COLLECTION", "change_chunks"),
		ChangeDateField:           getEnv("CHANGE_DATE_FIELD", "date"),
		CodeCollection:            getEnv("CODE_COLLECTION", "code_chunks"),
		DocCollection:             getEnv("DOC_COLLECTION", "doc_chunks"),
		AppProfileDir:             getEnv("APP_PROFILE_DIR", "/etc/app-prof"),
		AppProfileFiles: map[string]string{
			"github.com/Blazemeter/bzm-crane":  "bzm-crane.txt",
			"github.com/Blazemeter/taurus":     "taurus.txt",
			"github.com/Blazemeter/helm-crane": "helm-crane.txt",
			"github.com/Blazemeter/bzm-mcp":    "bzm-mcp.txt",
			"github.com/Blazemeter/sv-mcp":     "sv-mcp.txt",
		},
	}, nil
}

// parseEmbedServiceType requires EMBED_SERVICE_TYPE in "<type>:<host>" format; no default client type.
func parseEmbedServiceType(raw string) (string, string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", "", fmt.Errorf("EMBED_SERVICE_TYPE must be set (format: <type>:<host>)")
	}

	parts := strings.SplitN(raw, ":", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("EMBED_SERVICE_TYPE %q must be in format <type>:<host>", raw)
	}

	clientType := strings.TrimSpace(parts[0])
	host := strings.TrimSpace(parts[1])
	if clientType == "" || host == "" {
		return "", "", fmt.Errorf("EMBED_SERVICE_TYPE %q must have non-empty type and host", raw)
	}

	return clientType, host, nil
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

func normalizeListenAddr(port string) string {
	port = strings.TrimSpace(port)
	if port == "" {
		return ":8080"
	}
	if strings.HasPrefix(port, ":") {
		return port
	}
	return ":" + port
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvDuration(key string, fallback time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}

	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}

	return d
}

// getEnvFloat32 parses key as a float32, e.g. a Qdrant cosine-similarity score threshold (0 disables it).
func getEnvFloat32(key string, fallback float32) float32 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}

	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return fallback
	}

	return float32(f)
}

func getEnvBool(key string, fallback bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}

	b, err := strconv.ParseBool(v)
	if err != nil {
		return fallback
	}

	return b
}

func getEnvInt(key string, fallback int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}

	i, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}

	return i
}
