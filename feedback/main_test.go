package main

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"
)

func TestFeedbackServiceHTTP(t *testing.T) {
	app := newApp(processFeedback, zerolog.Nop())
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
		response, err := client.Post(baseURL+"/api/v1/rag-go/feedback", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		var result map[string]string
		err = json.NewDecoder(response.Body).Decode(&result)
		response.Body.Close()
		if err != nil {
			t.Fatal(err)
		}
		if response.StatusCode != http.StatusNotImplemented || result["error"] != ErrFeedbackNotImplemented.Error() {
			t.Fatalf("unconfigured persistence status=%d body=%v", response.StatusCode, result)
		}
	}
}
