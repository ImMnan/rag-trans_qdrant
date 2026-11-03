package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/gofiber/fiber/v3"
	"github.com/rs/zerolog"
)

// OrcaRequest mirrors the RAG HTTP request, including the subset used by generate-doc.
// Keep these wire types independent of Orca's pipeline and downstream dependencies.
type OrcaRequest struct {
	QueryText  string `json:"query_text"`
	RepoID     string `json:"repo_id,omitempty"`
	RepoName   string `json:"repo_name,omitempty"`
	Type       string `json:"type,omitempty"`
	Limit      int    `json:"limit,omitempty"`
	TokenLimit int    `json:"token_limit,omitempty"`
	Component  string `json:"component,omitempty"`
	FromDate   string `json:"from_date,omitempty"`
	ToDate     string `json:"to_date,omitempty"`
}

// OrcaResponse mirrors pipeline.Response for standard, direct, and generate-doc.
// For generate-doc, Answer contains a JSON-encoded document result, not an object.
type OrcaResponse struct {
	Answer  string            `json:"answer"`
	Type    string            `json:"type"`
	Sources map[string]int    `json:"sources"`
	Meta    *OrcaResponseMeta `json:"meta"`
}

type OrcaResponseMeta struct {
	RepoID                   string `json:"repo_id"`
	Component                string `json:"component,omitempty"`
	QueryText                string `json:"query_text"`
	ContextTruncationEnabled bool   `json:"context_truncation_enabled"`
	ContextTruncationActive  bool   `json:"context_truncation_active"`
}

type Feedback struct {
	Rating  string `json:"rating"`
	Comment string `json:"comment,omitempty"`
}

type FeedbackSubmission struct {
	Workflow  string        `json:"workflow"`
	RequestID string        `json:"request_id,omitempty"`
	Request   *OrcaRequest  `json:"request"`
	Response  *OrcaResponse `json:"response"`
	Feedback  *Feedback     `json:"feedback"`
}

type FeedbackProcessor func(context.Context, FeedbackSubmission) error

var ErrFeedbackNotImplemented = errors.New("feedback persistence is not implemented")

func processFeedback(_ context.Context, _ FeedbackSubmission) error {
	// Implement database persistence here; only return nil after processing succeeds.
	return ErrFeedbackNotImplemented
}

func Register(app *fiber.App, process FeedbackProcessor, log zerolog.Logger) {
	app.Post("/api/v1/rag-go/feedback", func(c fiber.Ctx) error {
		var submission FeedbackSubmission
		if err := json.Unmarshal(c.Body(), &submission); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid JSON request body"})
		}
		if err := submission.validate(); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
		}
		if err := process(c.Context(), submission); err != nil {
			log.Error().Err(err).
				Str("workflow", submission.Workflow).
				Str("request_id", submission.RequestID).
				Msg("feedback processing failed")
			if errors.Is(err, ErrFeedbackNotImplemented) {
				return c.Status(fiber.StatusNotImplemented).JSON(fiber.Map{"error": ErrFeedbackNotImplemented.Error()})
			}
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "feedback processing failed"})
		}
		log.Info().
			Str("workflow", submission.Workflow).
			Str("request_id", submission.RequestID).
			Str("rating", submission.Feedback.Rating).
			Msg("feedback processed")
		return c.JSON(fiber.Map{"status": "received"})
	})
	app.Get("/health", func(c fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok", "service": "orca-feedback"})
	})
}

func (s FeedbackSubmission) validate() error {
	switch s.Workflow {
	case "standard", "direct", "generate-doc":
	default:
		return errors.New("workflow must be standard, direct, or generate-doc")
	}
	if s.Request == nil || s.Response == nil {
		return errors.New("request and response objects are required")
	}
	if strings.TrimSpace(s.Request.QueryText) == "" {
		return errors.New("request.query_text is required")
	}
	if strings.TrimSpace(s.Request.RepoID) == "" && strings.TrimSpace(s.Request.Component) == "" {
		return errors.New("request.repo_id or request.component is required")
	}
	if s.Workflow != "generate-doc" {
		requestType := strings.ToLower(strings.TrimSpace(s.Request.Type))
		// Orca treats a request with an omitted type as a direct request.
		if requestType == "" {
			requestType = "direct"
		}
		if requestType != s.Workflow {
			return fmt.Errorf("request.type must match workflow %q", s.Workflow)
		}
		if s.Response.Type != s.Request.Type {
			return errors.New("response.type must match request.type")
		}
	}
	if strings.TrimSpace(s.Response.Answer) == "" {
		return errors.New("response.answer is required")
	}
	if s.Response.Sources == nil || s.Response.Meta == nil {
		return errors.New("response.sources and response.meta objects are required")
	}
	if s.Feedback == nil || (s.Feedback.Rating != "up" && s.Feedback.Rating != "down") {
		return errors.New("feedback.rating must be up or down")
	}
	return nil
}
