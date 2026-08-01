package handler

import (
	"github.com/gofiber/fiber/v2"
	"github.com/rs/zerolog"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
)

// Register mounts all routes onto the Fiber app.
func Register(app *fiber.App, ragPipe *pipeline.RAGPipeline, docPipe *pipeline.DOCPipeline, log zerolog.Logger) {
	rp := &ragHandler{pipe: ragPipe, log: log}
	dp := &docHandler{pipe: docPipe, log: log}
	app.Post("/api/v1/rag-go", rp.handleRAG)
	app.Get("/health", handleHealth)
	app.Post("/api/v1/rag-go/generate-doc", dp.handleDocGenerate)
}

type ragHandler struct {
	pipe *pipeline.RAGPipeline
	log  zerolog.Logger
}

type docHandler struct {
	pipe *pipeline.DOCPipeline
	log  zerolog.Logger
}

// RAGRequest mirrors the JSON body expected by the API.
type RAGRequest struct {
	QueryText  string `json:"query_text"`
	RepoID     string `json:"repo_id",omitempty"`
	RepoName   string `json:"repo_name,omitempty"`
	Type       string `json:"type"`
	Limit      int    `json:"limit"`
	TokenLimit int    `json:"token_limit"`
	Component  string `json:"component,omitempty"` // optional, if repoID is specified.
}

func (rp *ragHandler) handleRAG(c *fiber.Ctx) error {
	var req RAGRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}

	if req.QueryText == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "query_text is required"})
	}
	if req.RepoID == "" && req.Component == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "repo_id or component name is required"})
	}
	if req.Limit <= 0 {
		req.Limit = 5
	}

	rp.log.Info().
		Str("repo_id", req.RepoID).
		Str("type", req.Type).
		Str("component", req.Component).
		Int("limit", req.Limit).
		Int("token_limit", req.TokenLimit).
		Msg("rag request received")

	result, err := rp.pipe.Execute(c.Context(), pipeline.Request{
		QueryText:  req.QueryText,
		RepoID:     req.RepoID,
		Type:       req.Type,
		Limit:      req.Limit,
		TokenLimit: req.TokenLimit,
		Component:  req.Component,
	})
	if err != nil {
		rp.log.Error().Err(err).Str("repo_id", req.RepoID).Msg("pipeline execution failed")
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "pipeline failed"})
	}

	return c.JSON(result)
}

func (dp *docHandler) handleDocGenerate(c *fiber.Ctx) error {
	var req RAGRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}

	if req.QueryText == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "query_text is required"})
	}
	if req.RepoID == "" && req.Component == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "repo_id or component name is required"})
	}
	if req.Limit <= 0 {
		req.Limit = 5
	}

	dp.log.Info().
		Str("repo_id", req.RepoID).
		Str("type", req.Type).
		Int("limit", req.Limit).
		Msg("doc generate/update request received")

	result, err := dp.pipe.Execute(c.Context(), pipeline.Request{
		QueryText:  req.QueryText,
		RepoID:     req.RepoID,
		Type:       req.Type,
		Limit:      req.Limit,
		TokenLimit: req.TokenLimit,
		Component:  req.Component,
	})
	if err != nil {
		dp.log.Error().Err(err).Str("repo_id", req.RepoID).Str("component", req.Component).Msg("pipeline execution failed")
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "pipeline failed"})
	}

	return c.JSON(result)
}

func handleHealth(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{"status": "ok", "service": "rag-go"})
}
