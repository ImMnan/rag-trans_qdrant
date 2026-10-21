package handler

import (
	"context"
	"fmt"
	"io"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gofiber/fiber/v3"
	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
	"github.com/rs/zerolog"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type failedStandardRetrieval struct {
	pipeline.QdrantQuerier
	err error
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
