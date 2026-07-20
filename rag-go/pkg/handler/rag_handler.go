package handler

import (
	"github.com/gofiber/fiber/v2"
	"github.com/rs/zerolog"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
)

// Register mounts all routes onto the Fiber app.
func Register(app *fiber.App, pipe *pipeline.RAGPipeline, log zerolog.Logger) {
	h := &ragHandler{pipe: pipe, log: log}

	app.Post("/api/v1/rag-go", h.handleRAG)
	app.Get("/health", handleHealth)
}

type ragHandler struct {
	pipe *pipeline.RAGPipeline
	log  zerolog.Logger
}

// RAGRequest mirrors the JSON body expected by the API.
type RAGRequest struct {
	QueryText string `json:"query_text"`
	RepoID    string `json:"repo_id"`
	Type      string `json:"type"`
	Limit     int    `json:"limit"`
}

func (h *ragHandler) handleRAG(c *fiber.Ctx) error {
	var req RAGRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}

	if req.QueryText == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "query_text is required"})
	}
	if req.RepoID == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "repo_id is required"})
	}
	if req.Limit <= 0 {
		req.Limit = 5
	}

	h.log.Info().
		Str("repo_id", req.RepoID).
		Str("type", req.Type).
		Int("limit", req.Limit).
		Msg("rag request received")

	result, err := h.pipe.Execute(c.Context(), pipeline.Request{
		QueryText: req.QueryText,
		RepoID:    req.RepoID,
		Type:      req.Type,
		Limit:     req.Limit,
	})
	if err != nil {
		h.log.Error().Err(err).Str("repo_id", req.RepoID).Msg("pipeline execution failed")
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "pipeline failed"})
	}

	return c.JSON(result)
}

func handleHealth(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{"status": "ok", "service": "rag-go"})
}
