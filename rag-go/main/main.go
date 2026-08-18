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
	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Str("service", "orca").Logger()

	// --- Config from env ---
	cfg := loadConfig()
	orcaVersion := "0.7+"
	log.Info().
		Str("port", cfg.FiberPort).
		Str("qdrant_host", cfg.QdrantHost).
		Str("vllm_host", cfg.VLLMHost).
		Str("embed_client_type", cfg.EmbedClientType).
		Str("embed_host", cfg.EmbedHost).
		Str("Orca version", orcaVersion).
		Str("maintainer", "https://github.com/ImMnan").
		Msg("starting Orca service")

	// --- Clients ---
	qdrantClient := qdrant.NewClient(cfg.QdrantHost, log.Logger)
	embedClient := embedder.NewClientFromType(cfg.EmbedClientType, buildHTTPURL(cfg.EmbedHost), log.Logger)
	log.Info().Str("url", buildHTTPURL(cfg.VLLMHost)).Msg("vllm transport: http")
	vllmClient := vllm.NewHTTPClient(buildHTTPURL(cfg.VLLMHost), cfg.ModelName, cfg.VLLMTimeout, log.Logger)

	// --- Pipeline ---
	pipe := pipeline.New(qdrantClient, vllmClient, embedClient, cfg.ChangeCollection, cfg.CodeCollection, cfg.ChangeDateField)
	docPipe := pipeline.NewDoc(qdrantClient, vllmClient, embedClient, cfg.ChangeCollection, cfg.CodeCollection, cfg.DocCollection, cfg.GenDocCollection)

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
