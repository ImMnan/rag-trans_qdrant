package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/embedder"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/handler"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/qdrant"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/vllm"
)

func main() {
	// --- Logging ---
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Str("service", "rag-go").Logger()

	// --- Config from env ---
	cfg := loadConfig()

	log.Info().
		Str("port", cfg.FiberPort).
		Str("qdrant_host", cfg.QdrantHost).
		Str("vllm_host", cfg.VLLMHost).
		Str("embed_host", cfg.EmbedHost).
		Msg("starting rag-go service")

	// --- Clients ---
	qdrantClient := qdrant.NewClient(cfg.QdrantHost, log.Logger)
	embedClient := embedder.NewClient(buildHTTPURL(cfg.EmbedHost), log.Logger)
	log.Info().Str("url", buildHTTPURL(cfg.VLLMHost)).Msg("vllm transport: http")
	vllmClient := vllm.NewHTTPClient(buildHTTPURL(cfg.VLLMHost), cfg.ModelName, log.Logger)

	// --- Pipeline ---
	pipe := pipeline.New(qdrantClient, vllmClient, embedClient, cfg.ChangeCollection, cfg.CodeCollection)

	// --- Fiber app ---
	app := fiber.New(fiber.Config{
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second, // vLLM inference can be slow
		IdleTimeout:  60 * time.Second,
	})

	handler.Register(app, pipe, log.Logger)

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
