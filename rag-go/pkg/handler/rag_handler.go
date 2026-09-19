package handler

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/rs/zerolog"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
)

// Register mounts all routes onto the Fiber app.
func Register(app *fiber.App, ragPipe *pipeline.RAGPipeline, docPipe *pipeline.DOCPipeline, log zerolog.Logger) {
	rp := &ragHandler{pipe: ragPipe, log: log}
	dp := &docHandler{pipe: docPipe, log: log}
	app.Post("/api/v1/rag-go", rp.handleRAG)
	app.Get("/health", handleHealth)
	app.Post("/api/v1/rag-go/generate-doc", dp.handleRAG)
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
	RepoID     string `json:"repo_id,omitempty"`
	RepoName   string `json:"repo_name,omitempty"`
	Type       string `json:"type"`
	Limit      int    `json:"limit"`
	TokenLimit int    `json:"token_limit"`
	Component  string `json:"component,omitempty"` // optional, if repoID is specified.
	FromDate   string `json:"from_date,omitempty"` // YYYY-MM-DD, optional; if provided, filters chunks to this date or later.
	ToDate     string `json:"to_date,omitempty"`   // YYYY-MM-DD, optional; if provided, filters chunks to this date or earlier.
}

// DocGenerateRequest contains the fields supported by the document workflow.
type DocGenerateRequest struct {
	QueryText  string `json:"query_text"`
	RepoID     string `json:"repo_id,omitempty"`
	RepoName   string `json:"repo_name,omitempty"`
	Limit      int    `json:"limit"`
	TokenLimit int    `json:"token_limit"`
	Component  string `json:"component,omitempty"`
}

func (rp *ragHandler) handleRAG(c fiber.Ctx) error {
	var req RAGRequest
	if err := c.Bind().Body(&req); err != nil {
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
	if req.FromDate != "" {
		if _, err := time.Parse("2006-01-02", req.FromDate); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": fmt.Sprintf("from_date must use YYYY-MM-DD: %v", err)})
		}
	}
	if req.ToDate != "" {
		if _, err := time.Parse("2006-01-02", req.ToDate); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": fmt.Sprintf("to_date must use YYYY-MM-DD: %v", err)})
		}
	}
	if req.FromDate != "" && req.ToDate != "" && strings.Compare(req.FromDate, req.ToDate) > 0 {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "from_date must be on or before to_date"})
	}

	// Default to last 30 days if no explicit date range provided.
	if req.FromDate == "" && req.ToDate == "" {
		now := time.Now()
		req.FromDate = now.AddDate(0, 0, -30).Format("2006-01-02")
		req.ToDate = now.Format("2006-01-02")
	}
	stream, err := streamRequested(c)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	rp.log.Info().
		Str("repo_id", req.RepoID).
		Str("type", req.Type).
		Str("component", req.Component).
		Int("limit", req.Limit).
		Int("token_limit", req.TokenLimit).
		Str("from_date", req.FromDate).
		Str("to_date", req.ToDate).
		Msg("rag request received")

	pipelineRequest := pipeline.Request{
		QueryText:  req.QueryText,
		RepoID:     req.RepoID,
		Type:       req.Type,
		Limit:      req.Limit,
		TokenLimit: req.TokenLimit,
		Component:  req.Component,
		FromDate:   req.FromDate,
		ToDate:     req.ToDate,
	}
	if stream {
		requestContext := c.Context()
		return streamPipelineResponse(c, func(progress pipeline.ProgressReporter) (*pipeline.Response, error) {
			pipelineRequest.Progress = progress
			return rp.pipe.Execute(requestContext, pipelineRequest)
		})
	}

	result, err := rp.pipe.Execute(c.Context(), pipelineRequest)
	if err != nil {
		rp.log.Error().Err(err).Str("repo_id", req.RepoID).Msg("pipeline execution failed")
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "pipeline failed"})
	}

	return c.JSON(result)
}

func (dp *docHandler) handleRAG(c fiber.Ctx) error {
	var req DocGenerateRequest
	if err := c.Bind().Body(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}

	if req.QueryText == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "query_text is required"})
	}
	if req.RepoID == "" && req.Component == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "repo_id or component name is required"})
	}
	if req.Limit <= 0 {
		req.Limit = 25
	}
	stream, err := streamRequested(c)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	dp.log.Info().
		Str("repo_id", req.RepoID).
		Str("component", req.Component).
		Int("limit", req.Limit).
		Msg("doc generate/update request received")

	pipelineRequest := pipeline.Request{
		QueryText:  req.QueryText,
		RepoID:     req.RepoID,
		Limit:      req.Limit,
		TokenLimit: req.TokenLimit,
		Component:  req.Component,
	}
	if stream {
		requestContext := c.Context()
		return streamPipelineResponse(c, func(progress pipeline.ProgressReporter) (*pipeline.Response, error) {
			pipelineRequest.Progress = progress
			return dp.pipe.Execute(requestContext, pipelineRequest)
		})
	}

	result, err := dp.pipe.Execute(c.Context(), pipelineRequest)
	if err != nil {
		dp.log.Error().Err(err).Str("repo_id", req.RepoID).Str("component", req.Component).Msg("pipeline execution failed")
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "pipeline failed"})
	}

	return c.JSON(result)
}

func handleHealth(c fiber.Ctx) error {
	return c.JSON(fiber.Map{"status": "ok", "service": "orca"})
}

func streamRequested(c fiber.Ctx) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(c.Query("stream"))) {
	case "", "false":
		return false, nil
	case "true":
		return true, nil
	default:
		return false, fmt.Errorf("stream must be true or false")
	}
}

func streamPipelineResponse(c fiber.Ctx, execute func(pipeline.ProgressReporter) (*pipeline.Response, error)) error {
	c.Set(fiber.HeaderContentType, "text/event-stream")
	c.Set(fiber.HeaderCacheControl, "no-cache")
	c.Set("X-Accel-Buffering", "no")

	return c.SendStreamWriter(func(w *bufio.Writer) {
		writeSSE(w, "progress", pipeline.ProgressEvent{
			Stage:   "request_received",
			Message: "Request received",
			Percent: 0,
		})

		result, err := execute(func(event pipeline.ProgressEvent) {
			writeSSE(w, "progress", event)
		})
		if err != nil {
			writeSSE(w, "error", fiber.Map{"error": "pipeline failed"})
			return
		}

		writeSSE(w, "complete", result)
	})
}

func writeSSE(w *bufio.Writer, event string, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	_ = w.Flush()
}
