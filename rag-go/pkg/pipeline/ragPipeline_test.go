package pipeline

import (
	"context"
	"errors"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type failingQdrant struct {
	QdrantQuerier
	err            error
	failCollection string
}

func (q *failingQdrant) QueryStandardAll(context.Context, string, string, string, string, string, string) ([]string, error) {
	return []string{"partial context"}, q.err
}

func (q *failingQdrant) Query(_ context.Context, collection string, _ []float32, _ []uint32, _ []float32, _, _ string, _ int) ([]string, error) {
	if q.failCollection != "" && q.failCollection != collection {
		return []string{"retrieved context"}, nil
	}
	return []string{"partial context"}, q.err
}

type countingLLM struct {
	calls int
}

func (llm *countingLLM) Complete(context.Context, []Message, int) (string, error) {
	llm.calls++
	return "unexpected answer", nil
}

type testEmbedder struct{}

func (testEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{1}, nil
}

func TestRAGRetrievalFailureStopsGeneration(t *testing.T) {
	for _, requestType := range []string{"standard", "code"} {
		for _, code := range []codes.Code{codes.ResourceExhausted, codes.Unavailable} {
			t.Run(requestType+"/"+code.String(), func(t *testing.T) {
				cause := status.Error(code, "retrieval failed")
				llm := &countingLLM{}
				pipe := New(&failingQdrant{err: cause}, llm, testEmbedder{}, nil, false, 1, false, "changes", "code", "date", "", nil)
				var stages []string
				response, err := pipe.Execute(context.Background(), Request{
					Type: requestType,
					Progress: func(event ProgressEvent) {
						stages = append(stages, event.Stage)
					},
				})
				if response != nil || !errors.Is(err, cause) {
					t.Fatalf("expected original retrieval error and no response, got %v, %v", response, err)
				}
				if llm.calls != 0 {
					t.Fatalf("LLM called %d times after retrieval failure", llm.calls)
				}
				for _, stage := range stages {
					if stage != "embedding_complete" {
						t.Fatalf("unexpected progress after retrieval failure: %s", stage)
					}
				}
			})
		}
	}
}

func TestDOCRetrievalFailureStopsGeneration(t *testing.T) {
	for _, collection := range []string{"code", "docs"} {
		t.Run(collection, func(t *testing.T) {
			cause := status.Error(codes.ResourceExhausted, "retrieval failed")
			llm := &countingLLM{}
			pipe := NewDoc(&failingQdrant{err: cause, failCollection: collection}, llm, testEmbedder{}, nil, false, 1, false, "code", "docs", "", nil)
			response, err := pipe.Execute(context.Background(), Request{})
			if response != nil || !errors.Is(err, cause) {
				t.Fatalf("expected original retrieval error and no response, got %v, %v", response, err)
			}
			if llm.calls != 0 {
				t.Fatalf("LLM called %d times after retrieval failure", llm.calls)
			}
		})
	}
}
