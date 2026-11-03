package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rs/zerolog"
)

func TestFeedbackRejectsInvalidPayloads(t *testing.T) {
	for _, tt := range []struct {
		name    string
		body    string
		message string
	}{
		{"malformed JSON", `{`, "invalid JSON"},
		{"trailing JSON", standardPayload + `{}`, "invalid JSON"},
		{"null", `null`, "workflow"},
		{"array", `[]`, "invalid JSON"},
		{"empty envelope", `{}`, "workflow"},
		{"unknown workflow", strings.Replace(standardPayload, `"workflow": "standard"`, `"workflow": "other"`, 1), "workflow"},
		{"missing request", `{"workflow":"standard","response":{}}`, "request and response"},
		{"null response", `{"workflow":"standard","request":{},"response":null}`, "request and response"},
		{"empty request", replacePayload(t, func(s *FeedbackSubmission) { s.Request = &OrcaRequest{} }), "request.query_text"},
		{"blank query", replacePayload(t, func(s *FeedbackSubmission) { s.Request.QueryText = " " }), "request.query_text"},
		{"missing scope", replacePayload(t, func(s *FeedbackSubmission) { s.Request.RepoID = ""; s.Request.Component = "" }), "request.repo_id"},
		{"mismatched request type", replacePayload(t, func(s *FeedbackSubmission) { s.Request.Type = "direct" }), "request.type"},
		{"mismatched response type", replacePayload(t, func(s *FeedbackSubmission) { s.Response.Type = "direct" }), "response.type"},
		{"empty response", replacePayload(t, func(s *FeedbackSubmission) { s.Response = &OrcaResponse{Type: "standard"} }), "response.answer"},
		{"missing sources", replacePayload(t, func(s *FeedbackSubmission) { s.Response.Sources = nil }), "response.sources"},
		{"missing meta", replacePayload(t, func(s *FeedbackSubmission) { s.Response.Meta = nil }), "response.meta"},
		{"missing feedback", replacePayload(t, func(s *FeedbackSubmission) { s.Feedback = nil }), "feedback.rating"},
		{"empty rating", replacePayload(t, func(s *FeedbackSubmission) { s.Feedback.Rating = "" }), "feedback.rating"},
		{"invalid rating", replacePayload(t, func(s *FeedbackSubmission) { s.Feedback.Rating = "yes" }), "feedback.rating"},
		{"wrong request shape", strings.Replace(standardPayload, `"request": {`, `"request": [{"invalid":true}], "unused": {`, 1), "invalid JSON"},
		{"wrong response shape", strings.Replace(standardPayload, `"sources": {`, `"sources": "invalid", "unused": {`, 1), "invalid JSON"},
		{"doc answer object", strings.Replace(directPayload, `"answer": "Follow these steps.\nRun the deployment."`, `"answer": {"title":"Deploy"}`, 1), "invalid JSON"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			app := newApp(func(context.Context, FeedbackSubmission) error {
				called = true
				return nil
			}, zerolog.Nop())
			request := httptest.NewRequest(http.MethodPost, "/api/v1/rag-go/feedback", strings.NewReader(tt.body))
			request.Header.Set("Content-Type", "application/json")
			response, err := app.Test(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			var body map[string]string
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != http.StatusBadRequest || called || !strings.Contains(body["error"], tt.message) {
				t.Fatalf("status=%d called=%t body=%v", response.StatusCode, called, body)
			}
		})
	}
}

func replacePayload(t *testing.T, change func(*FeedbackSubmission)) string {
	t.Helper()
	var submission FeedbackSubmission
	if err := json.Unmarshal([]byte(standardPayload), &submission); err != nil {
		t.Fatal(err)
	}
	change(&submission)
	body, err := json.Marshal(submission)
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}

func TestFeedbackProcessingErrors(t *testing.T) {
	for _, tt := range []struct {
		name    string
		process FeedbackProcessor
		status  int
		message string
	}{
		{"not implemented", processFeedback, http.StatusNotImplemented, ErrFeedbackNotImplemented.Error()},
		{"database failure", func(context.Context, FeedbackSubmission) error {
			return errors.New("private database connection details")
		}, http.StatusInternalServerError, "feedback processing failed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			app := newApp(tt.process, zerolog.Nop())
			request := httptest.NewRequest(http.MethodPost, "/api/v1/rag-go/feedback", strings.NewReader(standardPayload))
			request.Header.Set("Content-Type", "application/json")
			response, err := app.Test(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			body, err := io.ReadAll(response.Body)
			if err != nil {
				t.Fatal(err)
			}
			if response.StatusCode != tt.status || string(body) != `{"error":"`+tt.message+`"}` {
				t.Fatalf("status=%d body=%s", response.StatusCode, body)
			}
		})
	}
}
