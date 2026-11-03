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
	log.Logger = zerolog.New(os.Stdout).With().Timestamp().Str("service", "orca-feedback").Logger()

	port := os.Getenv("FIBER_PORT")
	if port == "" {
		port = "8080"
	}
	log.Info().
		Str("port", port).
		Str("version", orcaVersion).
		Str("maintainer", "https://github.com/ImMnan").
		Msg("starting Orca Feedback service")

	app := newApp(processFeedback, log.Logger)

	// --- Graceful shutdown ---
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)
	defer signal.Stop(quit)

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
	if err := app.Listen(":" + port); err != nil {
		log.Fatal().Err(err).Msg("fiber listen error")
	}

	log.Info().Msg("server stopped")
}

func newApp(process FeedbackProcessor, logger zerolog.Logger) *fiber.App {
	app := fiber.New(fiber.Config{
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	})
	Register(app, process, logger)
	return app
}
