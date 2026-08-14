package qdrant

import (
	"context"
	"fmt"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"
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
	return c.query(ctx, collection, vector, repoID, limit, nil)
}

// QueryChanges filters only change history by its inclusive date window.
func (c *Client) QueryChanges(ctx context.Context, collection string, vector []float32, repoID string, limit int, fromDate, toDate, dateField string) ([]string, error) {
	from, err := time.Parse("2006-01-02", fromDate)
	if err != nil {
		return nil, fmt.Errorf("invalid from_date %q: %w", fromDate, err)
	}
	to, err := time.Parse("2006-01-02", toDate)
	if err != nil {
		return nil, fmt.Errorf("invalid to_date %q: %w", toDate, err)
	}
	if from.After(to) {
		return nil, fmt.Errorf("from_date %q is after to_date %q", fromDate, toDate)
	}
	if dateField == "" {
		dateField = "date"
	}

	return c.query(ctx, collection, vector, repoID, limit, qdrant.NewDatetimeRange(dateField, &qdrant.DatetimeRange{
		Gte: timestamppb.New(from.UTC()),
		Lte: timestamppb.New(to.UTC().Add(24*time.Hour - time.Nanosecond)),
	}))
}

func (c *Client) query(ctx context.Context, collection string, vector []float32, repoID string, limit int, dateCondition *qdrant.Condition) ([]string, error) {
	if c.points == nil {
		return nil, fmt.Errorf("qdrant client not initialised")
	}
	must := []*qdrant.Condition{qdrant.NewMatch("repo_id", repoID)}
	if dateCondition != nil {
		must = append(must, dateCondition)
	}

	resp, err := c.points.Query(ctx, &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(vector...),
		Filter: &qdrant.Filter{
			Must: must,
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

		// Extract chunk text — payload field is "chunk_text" in doc/code collections.
		var text string
		for _, field := range []string{"chunk_text", "text"} {
			if v, ok := hit.Payload[field]; ok {
				if sv := v.GetStringValue(); sv != "" {
					text = sv
					break
				}
			}
		}
		if text == "" {
			continue
		}

		// Prefix with file_path (falls back to doc_ref) so downstream
		// prompts can populate doc_ref without hallucinating a filename.
		source := ""
		for _, field := range []string{"file_path", "doc_ref"} {
			if v, ok := hit.Payload[field]; ok {
				if sv := v.GetStringValue(); sv != "" {
					source = sv
					break
				}
			}
		}
		if source != "" {
			text = "[source: " + source + "]\n" + text
		}

		chunks = append(chunks, text)
		c.log.Debug().
			Str("collection", collection).
			Str("point_id", pointID(hit.Id)).
			Float32("score", hit.Score).
			Int("order", len(chunks)-1).
			Msg("qdrant chunk retrieved")
	}

	c.log.Debug().
		Str("collection", collection).
		Int("hits", len(chunks)).
		Msg("qdrant query complete")

	return chunks, nil
}

func pointID(id *qdrant.PointId) string {
	if id == nil {
		return ""
	}
	if uuid := id.GetUuid(); uuid != "" {
		return uuid
	}
	return fmt.Sprintf("%d", id.GetNum())
}
