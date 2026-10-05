package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/embedder"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/handler"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/qdrant"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/vllm"
)

var orcaVersion = "dev"

func main() {
	// --- Logging ---
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	// Debug logs are emitted only when LOG_VERBOSE is set; default level is Info.
	if os.Getenv("LOG_VERBOSE") == "true" {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Str("service", "orca").Logger()

	// --- Config from env ---
	cfg, err := loadConfig()
	if err != nil {
		log.Fatal().Err(err).Msg("invalid configuration")
	}
	log.Info().
		Str("port", cfg.FiberPort).
		Str("qdrant_host", cfg.QdrantHost).
		Int("qdrant_max_call_recv_msg_size_kb", cfg.QdrantMaxCallRecvMsgSizeKB).
		Float32("qdrant_score_threshold", cfg.QdrantScoreThreshold).
		Bool("qdrant_mmr_enabled", cfg.QdrantMMREnabled).
		Float32("qdrant_mmr_lambda", cfg.QdrantMMRLambda).
		Int("qdrant_mmr_overfetch", cfg.QdrantMMROverfetch).
		Bool("qdrant_hybrid_enabled", cfg.QdrantHybridEnabled).
		Str("qdrant_dense_vector", cfg.QdrantDenseVectorName).
		Str("qdrant_sparse_vector", cfg.QdrantSparseVectorName).
		Int("qdrant_hybrid_prefetch", cfg.QdrantHybridPrefetch).
		Bool("context_truncation_enabled", cfg.ContextTruncationEnabled).
		Bool("rerank", cfg.RerankEnabled).
		Int("rerank_overfetch", cfg.RerankOverfetchMultiplier).
		Str("embed_client_type", cfg.EmbedClientType).
		Str("embed_host", cfg.EmbedHost).
		Str("vllm_host", cfg.VLLMHost).
		Str("Orca version", orcaVersion).
		Str("maintainer", "https://github.com/ImMnan").
		Msg("starting Orca service")

	// --- Clients ---
	qdrantClient := qdrant.NewClient(cfg.QdrantHost, cfg.QdrantMaxCallRecvMsgSizeKB, cfg.QdrantScoreThreshold, cfg.QdrantNeighborStitch, cfg.QdrantMMREnabled, cfg.QdrantMMRLambda, cfg.QdrantMMROverfetch, cfg.QdrantDenseVectorName, cfg.QdrantSparseVectorName, cfg.QdrantHybridPrefetch, log.Logger)
	embedClient := embedder.NewClientFromType(cfg.EmbedClientType, buildHTTPURL(cfg.EmbedHost), cfg.EmbedTimeout, log.Logger)
	var sparseEmbedder pipeline.SparseEmbedder
	if cfg.QdrantHybridEnabled {
		if s, ok := embedClient.(pipeline.SparseEmbedder); ok {
			sparseEmbedder = s
		} else {
			log.Warn().Str("embed_client_type", cfg.EmbedClientType).Msg("embed client has no sparse encoder, using dense-only retrieval")
		}
	}
	log.Info().Str("url", buildHTTPURL(cfg.VLLMHost)).Msg("vllm transport: http")
	vllmClient := vllm.NewHTTPClient(buildHTTPURL(cfg.VLLMHost), cfg.ModelName, cfg.VLLMTimeout, log.Logger)

	// --- Pipeline ---
	pipe := pipeline.New(qdrantClient, vllmClient, embedClient, embedClient, cfg.RerankEnabled, cfg.RerankOverfetchMultiplier, cfg.ContextTruncationEnabled, cfg.ChangeCollection, cfg.CodeCollection, cfg.ChangeDateField, cfg.AppProfileDir, cfg.AppProfileFiles).WithLogger(log.Logger).WithSparseEmbedder(sparseEmbedder)
	docPipe := pipeline.NewDoc(qdrantClient, vllmClient, embedClient, embedClient, cfg.RerankEnabled, cfg.RerankOverfetchMultiplier, cfg.ContextTruncationEnabled, cfg.CodeCollection, cfg.DocCollection, cfg.AppProfileDir, cfg.AppProfileFiles).WithLogger(log.Logger).WithSparseEmbedder(sparseEmbedder)

	// --- Fiber app ---
	app := fiber.New(fiber.Config{
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout, // vLLM inference can be slow
		IdleTimeout:  cfg.IdleTimeout,
	})

	handler.Register(app, pipe, docPipe, log.Logger)

	// --- Graceful shutdown ---
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)

	go func() {
		<-quit
		log.Info().Msg("shutdown signal received, draining requests...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := app.ShutdownWithContext(ctx); err != nil {
			log.Error().Err(err).Msg("error during graceful shutdown")
		}
	}()

	// --- Listen ---
	portNumber := normalizeListenAddr(cfg.FiberPort)
	if err := app.Listen(portNumber); err != nil {
		log.Fatal().Err(err).Msg("fiber listen error")
	}

	log.Info().Msg("server stopped")
}
