package handler

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
	"github.com/rs/zerolog"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type failedStandardRetrieval struct {
	pipeline.QdrantQuerier
	err error
}

type blockingRequest struct {
	pipeline.QdrantQuerier
	started chan struct{}
	release chan struct{}
}

func (b blockingRequest) wait(ctx context.Context) error {
	b.started <- struct{}{}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-b.release:
		return fmt.Errorf("test request released")
	}
}

func (b blockingRequest) QueryStandardAll(ctx context.Context, _, _, _, _, _, _ string) ([]string, error) {
	return nil, b.wait(ctx)
}

func (b blockingRequest) Embed(ctx context.Context, _ string) ([]float32, error) {
	return nil, b.wait(ctx)
}

func TestCancelActiveRequests(t *testing.T) {
	for _, path := range []string{"/api/v1/rag-go", "/api/v1/rag-go/generate-doc"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", path, stream), func(t *testing.T) {
				blocker := blockingRequest{started: make(chan struct{}, 1), release: make(chan struct{})}
				defer close(blocker.release)
				pipe := pipeline.New(blocker, nil, nil, nil, false, 1, false, "changes", "code", "date", "", nil)
				doc := pipeline.NewDoc(nil, nil, blocker, nil, false, 1, false, "code", "docs", "", nil)
				app := fiber.New()
				Register(app, pipe, doc, zerolog.Nop())
				id := uuid.NewString()
				newRequest := func() *http.Request {
					request := httptest.NewRequest("POST", fmt.Sprintf("%s?stream=%t", path, stream), strings.NewReader(`{"query_text":"summarize changes","repo_id":"repo","type":"standard"}`))
					request.Header.Set("Content-Type", "application/json")
					request.Header.Set("X-Request-ID", id)
					return request
				}
				type outcome struct {
					response *http.Response
					err      error
				}
				finished := make(chan outcome, 1)
				go func() {
					response, err := app.Test(newRequest(), fiber.TestConfig{Timeout: 5 * time.Second})
					finished <- outcome{response, err}
				}()
				select {
				case <-blocker.started:
				case <-time.After(3 * time.Second):
					t.Fatal("pipeline did not start")
				}
				duplicate, err := app.Test(newRequest())
				if err != nil {
					t.Fatal(err)
				}
				duplicate.Body.Close()
				if duplicate.StatusCode != fiber.StatusConflict {
					t.Fatalf("duplicate status = %d", duplicate.StatusCode)
				}
				cancel, err := app.Test(httptest.NewRequest("POST", "/api/v1/rag-go/requests/"+id+"/cancel", nil))
				if err != nil {
					t.Fatal(err)
				}
				cancelBody, _ := io.ReadAll(cancel.Body)
				cancel.Body.Close()
				if cancel.StatusCode != fiber.StatusAccepted || !strings.Contains(string(cancelBody), "cancellation_requested") {
					t.Fatalf("cancel status=%d body=%s", cancel.StatusCode, cancelBody)
				}
				select {
				case result := <-finished:
					if result.err != nil {
						t.Fatal(result.err)
					}
					defer result.response.Body.Close()
					body, err := io.ReadAll(result.response.Body)
					if err != nil {
						t.Fatal(err)
					}
					wantStatus := fiber.StatusConflict
					if stream {
						wantStatus = fiber.StatusOK
						if !strings.Contains(string(body), "event: cancelled") || !strings.Contains(string(body), `"stage":"request_received"`) {
							t.Fatalf("missing SSE events: %s", body)
						}
					}
					if result.response.StatusCode != wantStatus || result.response.Header.Get("X-Request-ID") != id || !strings.Contains(string(body), id) || !strings.Contains(string(body), `"status":"cancelled"`) {
						t.Fatalf("unexpected cancellation: status=%d body=%s", result.response.StatusCode, body)
					}
					if strings.Contains(string(body), "event: complete") || strings.Contains(string(body), "event: error") {
						t.Fatalf("unexpected terminal event: %s", body)
					}
				case <-time.After(3 * time.Second):
					t.Fatal("cancellation did not stop pipeline")
				}
				missing, err := app.Test(httptest.NewRequest("POST", "/api/v1/rag-go/requests/"+id+"/cancel", nil))
				if err != nil {
					t.Fatal(err)
				}
				missing.Body.Close()
				if missing.StatusCode != fiber.StatusNotFound {
					t.Fatalf("completed request still registered: %d", missing.StatusCode)
				}
			})
		}
	}
}

func TestRequestRegistryContract(t *testing.T) {
	registry := &requestRegistry{active: make(map[string]*activeRequest)}
	app := fiber.New()
	app.Post("/run", func(c fiber.Ctx) error {
		return registry.respond(c, c.Query("stream") == "true", func(context.Context, pipeline.ProgressReporter) (*pipeline.Response, error) {
			return &pipeline.Response{Answer: "done"}, nil
		})
	})
	app.Post("/cancel/:request_id", registry.handleCancel)
	for _, stream := range []bool{false, true} {
		response, err := app.Test(httptest.NewRequest("POST", fmt.Sprintf("/run?stream=%t", stream), nil))
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(response.Body)
		response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		id := response.Header.Get("X-Request-ID")
		if _, err := uuid.Parse(id); err != nil {
			t.Fatalf("invalid generated request ID %q", id)
		}
		if !strings.Contains(string(body), `"answer":"done"`) || (stream && !strings.Contains(string(body), id)) {
			t.Fatalf("unexpected success: %s", body)
		}
		registry.mu.Lock()
		remaining := len(registry.active)
		registry.mu.Unlock()
		if remaining != 0 {
			t.Fatalf("successful request leaked: %d", remaining)
		}
	}
	invalid := httptest.NewRequest("POST", "/run", nil)
	invalid.Header.Set("X-Request-ID", "not-a-uuid")
	for _, request := range []*http.Request{invalid, httptest.NewRequest("POST", "/cancel/not-a-uuid", nil)} {
		response, err := app.Test(request)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != fiber.StatusBadRequest {
			t.Fatalf("invalid ID status=%d", response.StatusCode)
		}
	}
	id := uuid.NewString()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	registry.active[id] = &activeRequest{cancel: cancel}
	for range 2 {
		response, err := app.Test(httptest.NewRequest("POST", "/cancel/"+id, nil))
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != fiber.StatusAccepted || ctx.Err() != context.Canceled {
			t.Fatalf("repeated cancellation status=%d err=%v", response.StatusCode, ctx.Err())
		}
	}
}

func (q failedStandardRetrieval) QueryStandardAll(context.Context, string, string, string, string, string, string) ([]string, error) {
	return nil, q.err
}

func TestRetrievalFailureReachesUI(t *testing.T) {
	testCases := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "oversized response",
			err:  status.Error(codes.ResourceExhausted, "grpc: received message larger than max (12447785 vs. 4194304)"),
			want: "pipeline failed: retrieved context exceeds the Qdrant gRPC response size limit; no answer was generated",
		},
		{name: "unavailable", err: status.Error(codes.Unavailable, "private connection details"), want: "pipeline failed"},
		{name: "other resource exhaustion", err: status.Error(codes.ResourceExhausted, "private quota details"), want: "pipeline failed"},
	}
	for _, testCase := range testCases {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", testCase.name, stream), func(t *testing.T) {
				cause := fmt.Errorf("qdrant standard scroll: %w", testCase.err)
				pipe := pipeline.New(failedStandardRetrieval{err: cause}, nil, nil, nil, false, 1, false, "changes", "code", "date", "", nil)
				app := fiber.New()
				Register(app, pipe, nil, zerolog.Nop())
				request := httptest.NewRequest("POST", fmt.Sprintf("/api/v1/rag-go?stream=%t", stream), strings.NewReader(`{"query_text":"summarize changes","repo_id":"repo","type":"standard"}`))
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
				wantBody := fmt.Sprintf(`{"error":%q}`, testCase.want)
				wantStatus := fiber.StatusInternalServerError
				if stream {
					wantStatus = fiber.StatusOK
					wantBody = "event: error\ndata: " + wantBody
				}
				if response.StatusCode != wantStatus || !strings.Contains(string(body), wantBody) {
					t.Fatalf("unexpected UI error: status=%d body=%s", response.StatusCode, body)
				}
				if strings.Contains(string(body), "event: complete") || strings.Contains(string(body), "messages_assembled") || strings.Contains(string(body), "private") {
					t.Fatalf("failure response includes success or private details: %s", body)
				}
			})
		}
	}
}
