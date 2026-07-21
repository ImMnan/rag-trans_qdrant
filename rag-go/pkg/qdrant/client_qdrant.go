package qdrant

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	retryDelay = 120 * time.Second
	maxRetries = 2
)

// Client wraps the Qdrant gRPC client.
type Client struct {
	points qdrant.PointsClient
	log    zerolog.Logger
}

// NewClient dials Qdrant over gRPC. On failure it retries once after 120s.
// If the connection cannot be established the process logs the error and
// continues — per-request calls will fail with a descriptive error.
func NewClient(host string, log zerolog.Logger) *Client {
	conn, err := dialWithRetry(host, log)
	if err != nil {
		log.Error().Err(err).Str("host", host).Msg("qdrant unavailable, requests will fail until connectivity is restored")
		return &Client{log: log}
	}
	return &Client{
		points: qdrant.NewPointsClient(conn),
		log:    log,
	}
}

func dialWithRetry(host string, log zerolog.Logger) (*grpc.ClientConn, error) {
	for attempt := 1; attempt <= maxRetries; attempt++ {
		conn, err := grpc.NewClient(host, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err == nil {
			log.Info().Str("host", host).Msg("qdrant gRPC connection established")
			return conn, nil
		}
		log.Error().Err(err).Int("attempt", attempt).Str("host", host).Msg("qdrant connection failed")
		if attempt < maxRetries {
			log.Info().Dur("retry_in", retryDelay).Msg("retrying qdrant connection")
			time.Sleep(retryDelay)
		}
	}
	return nil, fmt.Errorf("could not connect to qdrant at %s after %d attempts", host, maxRetries)
}

// Query retrieves text chunks from a collection filtered by repo_id.
func (c *Client) Query(ctx context.Context, collection string, vector []float32, repoID string, limit int) ([]string, error) {
	if c.points == nil {
		return nil, fmt.Errorf("qdrant client not initialised")
	}

	resp, err := c.points.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(vector...),
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatch("repo_id", repoID),
			},
		},
		Limit:       qdrant.PtrOf(uint64(limit)),
		WithPayload: qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant query %s: %w", collection, err)
	}

	chunks := make([]string, 0, len(resp.Result))
	for _, hit := range resp.Result {
		if hit.Payload == nil {
			continue
		}
		if v, ok := hit.Payload["text"]; ok {
			if sv := v.GetStringValue(); sv != "" {
				chunks = append(chunks, sv)
			}
		}
	}

	c.log.Debug().
		Str("collection", collection).
		Int("hits", len(chunks)).
		Msg("qdrant query complete")

	return chunks, nil
}
