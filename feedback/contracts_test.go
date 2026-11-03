package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/rs/zerolog"
)

const standardPayload = `{
	"workflow": "standard",
	"request_id": "8b8047e6-8363-4382-93e3-1ad99f75adf3",
	"request": {
		"query_text": "What changed?", "repo_id": "github.com/example/repo",
		"repo_name": "repo", "type": "standard", "limit": 5, "token_limit": 4096,
		"component": "api", "from_date": "2026-10-01", "to_date": "2026-10-31"
	},
	"response": {
		"answer": "The API was updated.", "type": "standard",
		"sources": {"change_chunks_retrieved": 5, "code_chunks_retrieved": 0, "change_batches": 1},
		"meta": {
			"repo_id": "github.com/example/repo", "component": "api", "query_text": "What changed?",
			"context_truncation_enabled": true, "context_truncation_active": false
		}
	},
	"feedback": {"rating": "up", "comment": "Accurate summary."}
}`

const directPayload = `{
	"workflow": "direct",
	"request": {"query_text": "How do I deploy?", "component": "api", "type": "direct"},
	"response": {
		"answer": "Follow these steps.\nRun the deployment.", "type": "direct",
		"sources": {"change_chunks_retrieved": 0, "code_chunks_retrieved": 3, "code_chunks_config": 2},
		"meta": {
			"repo_id": "", "component": "api", "query_text": "How do I deploy?",
			"context_truncation_enabled": true, "context_truncation_active": true
		}
	},
	"feedback": {"rating": "down"}
}`

const docPayload = `{
	"workflow": "generate-doc",
	"request": {
		"query_text": "Document deployment", "repo_id": "github.com/example/repo",
		"repo_name": "repo", "limit": 25, "token_limit": 8192, "component": "api"
	},
	"response": {
		"answer": "{\"status\":\"new_document_required\",\"coverage\":\"none\",\"confidence_overall\":0.6,\"topic\":\"Deployment\",\"title\":\"Deploy\",\"body_markdown\":\"# Deploy\\nSteps\",\"corrections\":[],\"missing_points\":[],\"matched_docs\":[],\"decision_reason\":\"No existing documentation\",\"warnings\":[]}",
		"type": "", "sources": {"code_chunks_retrieved": 5, "doc_chunks_retrieved": 0},
		"meta": {
			"repo_id": "github.com/example/repo", "component": "api", "query_text": "Document deployment",
			"context_truncation_enabled": false, "context_truncation_active": false
		}
	},
	"feedback": {"rating": "up", "comment": "Helpful document."}
}`

func TestFeedbackPreservesOrcaContracts(t *testing.T) {
	for _, tt := range []struct {
		name string
		body string
	}{
		{"standard", standardPayload},
		{"direct", directPayload},
		{"generate-doc", docPayload},
		{"direct with omitted type", strings.ReplaceAll(directPayload, `, "type": "direct"`, "")},
		{"document with ignored request type", strings.Replace(docPayload, `"limit": 25`, `"type": "direct", "limit": 25`, 1)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var received FeedbackSubmission
			calls := 0
			app := newApp(func(ctx context.Context, submission FeedbackSubmission) error {
				if ctx == nil {
					t.Fatal("missing processing context")
				}
				calls++
				received = submission
				return nil
			}, zerolog.Nop())
			request := httptest.NewRequest(http.MethodPost, "/api/v1/rag-go/feedback", strings.NewReader(tt.body))
			request.Header.Set("Content-Type", "application/json")
			response, err := app.Test(request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK || calls != 1 {
				t.Fatalf("status=%d processor calls=%d", response.StatusCode, calls)
			}
			var ack map[string]string
			if err := json.NewDecoder(response.Body).Decode(&ack); err != nil {
				t.Fatal(err)
			}
			if ack["status"] != "received" {
				t.Fatalf("unexpected acknowledgement: %v", ack)
			}
			encoded, err := json.Marshal(received)
			if err != nil {
				t.Fatal(err)
			}
			var original, forwarded any
			if err := json.Unmarshal([]byte(tt.body), &original); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(encoded, &forwarded); err != nil {
				t.Fatal(err)
			}
			// Orca always emits response.type, even when the request omitted it.
			if tt.name == "direct with omitted type" {
				original.(map[string]any)["response"].(map[string]any)["type"] = ""
			}
			if !reflect.DeepEqual(original, forwarded) {
				t.Fatalf("payload changed:\noriginal: %s\nforwarded: %s", tt.body, encoded)
			}
		})
	}
}
