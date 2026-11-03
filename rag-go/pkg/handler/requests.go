package handler

import (
	"context"
	"sync"

	"github.com/gofiber/fiber/v3"
	"github.com/google/uuid"

	"github.com/immnan/rag-trans_qdrant/rag-go/pkg/pipeline"
)

type activeRequest struct {
	cancel    context.CancelFunc
	cancelled bool
}

type requestRegistry struct {
	mu     sync.Mutex
	active map[string]*activeRequest
}

func (r *requestRegistry) respond(c fiber.Ctx, stream bool, execute func(context.Context, pipeline.ProgressReporter) (*pipeline.Response, error)) error {
	id := c.Get("X-Request-ID")
	if id == "" {
		id = uuid.NewString()
	} else {
		parsed, err := uuid.Parse(id)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "X-Request-ID must be a UUID"})
		}
		id = parsed.String()
	}
	ctx, cancel := context.WithCancel(c.Context())
	entry := &activeRequest{cancel: cancel}
	r.mu.Lock()
	if _, exists := r.active[id]; exists {
		r.mu.Unlock()
		cancel()
		return c.Status(fiber.StatusConflict).JSON(fiber.Map{"error": "request_id is already active"})
	}
	r.active[id] = entry
	r.mu.Unlock()
	c.Set("X-Request-ID", id)

	run := func(progress pipeline.ProgressReporter) (*pipeline.Response, error) {
		defer cancel()
		defer func() {
			r.mu.Lock()
			if r.active[id] == entry {
				delete(r.active, id)
			}
			r.mu.Unlock()
		}()
		result, err := execute(ctx, progress)
		r.mu.Lock()
		cancelled := entry.cancelled
		delete(r.active, id)
		r.mu.Unlock()
		if cancelled {
			return nil, context.Canceled
		}
		return result, err
	}
	if stream {
		return streamPipelineResponse(c, id, run)
	}
	result, err := run(nil)
	if err != nil {
		if isCancellation(err) {
			return c.Status(fiber.StatusConflict).JSON(fiber.Map{"request_id": id, "status": "cancelled"})
		}
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": pipelineErrorMessage(err)})
	}
	return c.JSON(result)
}

func (r *requestRegistry) handleCancel(c fiber.Ctx) error {
	parsed, err := uuid.Parse(c.Params("request_id"))
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "request_id must be a UUID"})
	}
	id := parsed.String()
	r.mu.Lock()
	entry, exists := r.active[id]
	if exists {
		entry.cancelled = true
		entry.cancel()
	}
	r.mu.Unlock()
	if !exists {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "request is not active", "request_id": id})
	}
	return c.Status(fiber.StatusAccepted).JSON(fiber.Map{"request_id": id, "status": "cancellation_requested"})
}
