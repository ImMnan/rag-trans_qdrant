package main

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"
)

func TestFeedbackServiceHTTP(t *testing.T) {
	directory := t.TempDir()
	process, err := newFileProcessor(directory)
	if err != nil {
		t.Fatal(err)
	}
	app := newApp(process, zerolog.Nop())
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	stopped := make(chan error, 1)
	go func() { stopped <- app.Listener(listener) }()
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := app.ShutdownWithContext(ctx); err != nil {
			t.Error(err)
		}
		select {
		case err := <-stopped:
			if err != nil {
				t.Error(err)
			}
		case <-ctx.Done():
			t.Error("server did not stop")
		}
	})
	baseURL := "http://" + listener.Addr().String()
	client := &http.Client{Timeout: 5 * time.Second}
	response, err := client.Get(baseURL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	var health map[string]string
	err = json.NewDecoder(response.Body).Decode(&health)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK || health["service"] != "orca-feedback" || health["status"] != "ok" {
		t.Fatalf("health status=%d body=%v", response.StatusCode, health)
	}
	for _, body := range []string{standardPayload, directPayload, docPayload} {
		request, err := http.NewRequest(http.MethodPost, baseURL+"/api/v1/rag-go/feedback", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("X-User-ID", "keycloak-subject")
		response, err := client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		var result map[string]string
		err = json.NewDecoder(response.Body).Decode(&result)
		response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusOK || result["status"] != "received" {
			t.Fatalf("persistence status=%d body=%v", response.StatusCode, result)
		}
		var expected FeedbackSubmission
		if err := json.Unmarshal([]byte(body), &expected); err != nil {
			t.Fatal(err)
		}
		expected.UserID = "keycloak-subject"
		saved, err := os.ReadFile(filepath.Join(directory, expected.RequestID+".json"))
		if err != nil {
			t.Fatal(err)
		}
		var actual FeedbackSubmission
		if err := json.Unmarshal(saved, &actual); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(expected, actual) {
			t.Fatalf("saved payload differs: %s", saved)
		}
		retry, err := http.NewRequest(http.MethodPost, baseURL+"/api/v1/rag-go/feedback", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		retry.Header.Set("Content-Type", "application/json")
		retry.Header.Set("X-User-ID", "different-user")
		duplicate, err := client.Do(retry)
		if err != nil {
			t.Fatal(err)
		}
		duplicate.Body.Close()
		if duplicate.StatusCode != http.StatusConflict {
			t.Fatalf("duplicate status=%d", duplicate.StatusCode)
		}
		unchanged, err := os.ReadFile(filepath.Join(directory, expected.RequestID+".json"))
		if err != nil || string(saved) != string(unchanged) {
			t.Fatalf("duplicate changed original file: %v", err)
		}
	}
}
