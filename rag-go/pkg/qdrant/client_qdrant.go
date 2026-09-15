package qdrant

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"
)

const (
	retryDelay         = 120 * time.Second
	maxRetries         = 2
	standardScrollPage = 256
)

// Client wraps the Qdrant gRPC client.
type Client struct {
	points         qdrant.PointsClient
	scoreThreshold float32
	neighborStitch bool
	mmrEnabled     bool
	mmrLambda      float32
	mmrOverfetch   int
	log            zerolog.Logger
}

// NewClient dials Qdrant over gRPC. On failure it retries once after 120s.
// If the connection cannot be established the process logs the error and
// continues — per-request calls will fail with a descriptive error.
// scoreThreshold <= 0 disables score filtering (all top-K hits are kept).
// neighborStitch expands each hit that carries file_path+chunk_index with its
// immediate previous/next chunk from the same file, to avoid mid-function truncation.
// mmrLambda controls relevance versus diversity (1 is relevance-only, 0 is diversity-only).
// mmrOverfetch is the candidate multiplier used before selecting the requested limit.
func NewClient(host string, scoreThreshold float32, neighborStitch, mmrEnabled bool, mmrLambda float32, mmrOverfetch int, log zerolog.Logger) *Client {
	if mmrLambda < 0 || mmrLambda > 1 {
		mmrLambda = 0.7
	}
	if mmrOverfetch < 1 {
		mmrOverfetch = 1
	}
	conn, err := dialWithRetry(host, log)
	if err != nil {
		log.Error().Err(err).Str("host", host).Msg("qdrant unavailable, requests will fail until connectivity is restored")
		return &Client{scoreThreshold: scoreThreshold, neighborStitch: neighborStitch, mmrEnabled: mmrEnabled, mmrLambda: mmrLambda, mmrOverfetch: mmrOverfetch, log: log}
	}
	return &Client{
		points:         qdrant.NewPointsClient(conn),
		scoreThreshold: scoreThreshold,
		neighborStitch: neighborStitch,
		mmrEnabled:     mmrEnabled,
		mmrLambda:      mmrLambda,
		mmrOverfetch:   mmrOverfetch,
		log:            log,
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
func (c *Client) Query(ctx context.Context, collection string, vector []float32, repoID, component string, limit int) ([]string, error) {
	return c.query(ctx, collection, vector, repoID, component, limit, nil)
}

// QueryStandard filters only change history by its inclusive date window.
func (c *Client) QueryStandard(ctx context.Context, collection string, vector []float32, repoID, component string, limit int, fromDate, toDate, dateField string) ([]string, error) {
	dateFilters, err := standardDateFilters(fromDate, toDate, dateField)
	if err != nil {
		return nil, err
	}

	return c.query(ctx, collection, vector, repoID, component, limit, dateFilters[0].condition)
}

// QueryStandardAll retrieves every change chunk in the requested inclusive date window.
// Standard answers compare the full change window against vector-ranked code chunks, so
// change chunks must not be reduced by query top-K, MMR, reranking, or score thresholds.
func (c *Client) QueryStandardAll(ctx context.Context, collection string, repoID, component string, fromDate, toDate, dateField string) ([]string, error) {
	if c.points == nil {
		return nil, fmt.Errorf("qdrant client not initialised")
	}

	dateFilters, err := standardDateFilters(fromDate, toDate, dateField)
	if err != nil {
		return nil, err
	}

	var chunks []string
	seen := make(map[string]struct{})
	for _, dateFilter := range dateFilters {
		must := filterConditions(repoID, component, dateFilter.condition)
		req := &qdrant.ScrollPoints{
			CollectionName: collection,
			Filter:         &qdrant.Filter{Must: must},
			Limit:          qdrant.PtrOf(uint32(standardScrollPage)),
			WithPayload:    qdrant.NewWithPayload(true),
		}

		for {
			resp, err := c.points.Scroll(ctx, req)
			if err != nil {
				return nil, fmt.Errorf("qdrant standard scroll %s using %s: %w", collection, dateFilter.field, err)
			}

			for _, point := range resp.Result {
				text := pointPayloadText(point)
				if text == "" {
					continue
				}
				key := pointPayloadKey(point, text)
				if _, exists := seen[key]; exists {
					continue
				}
				seen[key] = struct{}{}
				chunks = append(chunks, text)
			}

			req.Offset = resp.GetNextPageOffset()
			if req.Offset == nil {
				break
			}
		}
	}

	c.log.Info().
		Str("collection", collection).
		Int("hits", len(chunks)).
		Str("from_date", fromDate).
		Str("to_date", toDate).
		Str("date_field", effectiveDateField(dateField)).
		Msg("qdrant standard change scroll complete")

	return chunks, nil
}

type standardDateFilter struct {
	field     string
	condition *qdrant.Condition
}

func standardDateFilters(fromDate, toDate, dateField string) ([]standardDateFilter, error) {
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

	fields := orderedDateFields(dateField)
	filters := make([]standardDateFilter, 0, len(fields))
	for _, field := range fields {
		filters = append(filters, standardDateFilter{field: field, condition: dateConditionForField(field, from, to)})
	}
	return filters, nil
}

func effectiveDateField(dateField string) string {
	dateField = strings.TrimSpace(dateField)
	if dateField == "" {
		return "date"
	}
	return dateField
}

func orderedDateFields(dateField string) []string {
	fields := []string{effectiveDateField(dateField), "date", "date_short", "month"}
	seen := make(map[string]struct{}, len(fields))
	ordered := make([]string, 0, len(fields))
	for _, field := range fields {
		if _, exists := seen[field]; exists {
			continue
		}
		seen[field] = struct{}{}
		ordered = append(ordered, field)
	}
	return ordered
}

func dateConditionForField(field string, from, to time.Time) *qdrant.Condition {
	switch field {
	case "date_short":
		return qdrant.NewMatchKeywords(field, dateShortValues(from, to)...)
	case "month":
		return qdrant.NewMatchKeywords(field, monthValues(from, to)...)
	default:
		return qdrant.NewDatetimeRange(field, &qdrant.DatetimeRange{
			Gte: timestamppb.New(from.UTC()),
			Lte: timestamppb.New(to.UTC().Add(24*time.Hour - time.Nanosecond)),
		})
	}
}

func dateShortValues(from, to time.Time) []string {
	values := make([]string, 0, int(to.Sub(from).Hours()/24)+1)
	for day := from; !day.After(to); day = day.AddDate(0, 0, 1) {
		values = append(values, day.Format("2006-01-02"))
	}
	return values
}

func monthValues(from, to time.Time) []string {
	month := time.Date(from.Year(), from.Month(), 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(to.Year(), to.Month(), 1, 0, 0, 0, 0, time.UTC)
	values := []string{}
	for !month.After(end) {
		values = append(values, month.Format("2006-01"))
		month = month.AddDate(0, 1, 0)
	}
	return values
}

func (c *Client) query(ctx context.Context, collection string, vector []float32, repoID, component string, limit int, dateCondition *qdrant.Condition) ([]string, error) {
	if c.points == nil {
		return nil, fmt.Errorf("qdrant client not initialised")
	}
	must := filterConditions(repoID, component, dateCondition)

	candidateLimit := limit
	if c.mmrEnabled && candidateLimit > 0 {
		candidateLimit *= c.mmrOverfetch
	}

	req := &qdrant.QueryPoints{
		CollectionName: collection,
		Query:          qdrant.NewQuery(vector...),
		Filter: &qdrant.Filter{
			Must: must,
		},
		Limit:       qdrant.PtrOf(uint64(candidateLimit)),
		WithPayload: qdrant.NewWithPayload(true),
	}
	if c.mmrEnabled {
		req.WithVectors = qdrant.NewWithVectors(true)
	}
	if c.scoreThreshold > 0 {
		req.ScoreThreshold = qdrant.PtrOf(c.scoreThreshold)
	}

	resp, err := c.points.Query(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("qdrant query %s: %w", collection, err)
	}

	parsed := make([]parsedHit, 0, len(resp.Result))
	vectorHits := 0
	for _, hit := range resp.Result {
		if hit.Payload == nil {
			continue
		}

		// Extract chunk text — payload field is "chunk_text" in doc/code collections.
		text := payloadString(hit.Payload, "chunk_text", "text")
		if text == "" {
			continue
		}

		// file_path (falls back to doc_ref) so downstream prompts can populate
		// doc_ref without hallucinating a filename.
		source := payloadString(hit.Payload, "file_path", "doc_ref")
		chunkIndex, hasChunkIndex := payloadInt(hit.Payload, "chunk_index")

		candidate := parsedHit{
			text:          text,
			source:        source,
			chunkIndex:    chunkIndex,
			hasChunkIndex: hasChunkIndex,
			score:         hit.Score,
			vector:        denseVector(hit),
		}
		if len(candidate.vector) > 0 {
			vectorHits++
		}
		parsed = append(parsed, candidate)
		c.log.Debug().
			Str("collection", collection).
			Str("point_id", pointID(hit.Id)).
			Float32("score", hit.Score).
			Int("order", len(parsed)-1).
			Msg("qdrant chunk retrieved")
	}

	parsedBeforeMMR := len(parsed)
	if c.mmrEnabled {
		parsed = selectDiverse(parsed, vector, limit, c.mmrLambda)
	}

	if c.neighborStitch {
		c.stitchNeighborChunks(ctx, collection, repoID, component, parsed)
	}

	chunks := make([]string, 0, len(parsed))
	for _, p := range parsed {
		text := p.text
		if p.source != "" {
			text = "[source: " + p.source + "]\n" + text
		}
		chunks = append(chunks, text)
	}

	c.log.Info().
		Str("collection", collection).
		Int("hits", len(chunks)).
		Int("qdrant_candidates", len(resp.Result)).
		Int("parsed_candidates", parsedBeforeMMR).
		Int("vector_candidates", vectorHits).
		Bool("mmr_enabled", c.mmrEnabled).
		Float32("mmr_lambda", c.mmrLambda).
		Int("mmr_overfetch", c.mmrOverfetch).
		Bool("mmr_applied", c.mmrEnabled && parsedBeforeMMR > limit).
		Float32("score_threshold", c.scoreThreshold).
		Msg("qdrant query complete")

	return chunks, nil
}

func filterConditions(repoID, component string, conditions ...*qdrant.Condition) []*qdrant.Condition {
	var must []*qdrant.Condition
	if repoID != "" {
		must = append(must, qdrant.NewMatch("repo_id", repoID))
	}
	if component != "" {
		must = append(must, qdrant.NewMatch("component", component))
	}
	for _, condition := range conditions {
		if condition != nil {
			must = append(must, condition)
		}
	}
	return must
}

func pointPayloadText(point *qdrant.RetrievedPoint) string {
	if point == nil || point.Payload == nil {
		return ""
	}
	text := payloadString(point.Payload, "change_chunk", "chunk_text", "text")
	if text == "" {
		return ""
	}
	source := payloadString(point.Payload, "file_path", "doc_ref")
	if source != "" {
		text = "[source: " + source + "]\n" + text
	}
	return text
}

func pointPayloadKey(point *qdrant.RetrievedPoint, text string) string {
	if point != nil {
		if id := pointID(point.Id); id != "" {
			return id
		}
		if point.Payload != nil {
			return payloadString(point.Payload, "file_path", "doc_ref") + "|" + text
		}
	}
	return text
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

// parsedHit is a retrieved chunk before its final "[source: ...]" prefix is applied,
// carrying enough position info (file_path/chunk_index) for neighbor stitching.
type parsedHit struct {
	text          string
	source        string // file_path or doc_ref, used both for display and neighbor lookup
	chunkIndex    int64
	hasChunkIndex bool
	score         float32
	vector        []float32
}

func denseVector(hit *qdrant.ScoredPoint) []float32 {
	if hit == nil || hit.Vectors == nil || hit.Vectors.GetVector() == nil {
		return nil
	}
	return hit.Vectors.GetVector().GetDense().GetData()
}

// selectDiverse applies maximal marginal relevance and removes repeated chunk text.
// If Qdrant does not return usable vectors, it still performs stable exact deduplication.
func selectDiverse(hits []parsedHit, query []float32, limit int, lambda float32) []parsedHit {
	if limit <= 0 || len(hits) <= limit {
		return deduplicateHits(hits, limit)
	}
	remaining := deduplicateHits(hits, len(hits))
	selected := make([]parsedHit, 0, limit)
	for len(remaining) > 0 && len(selected) < limit {
		best := 0
		bestValue := float32(-math.MaxFloat32)
		for i, candidate := range remaining {
			value := lambda * candidate.score
			if len(candidate.vector) > 0 {
				value = lambda * cosine(query, candidate.vector)
				if len(selected) > 0 {
					maxSimilarity := float32(-1)
					for _, prior := range selected {
						if similarity := cosine(candidate.vector, prior.vector); similarity > maxSimilarity {
							maxSimilarity = similarity
						}
					}
					value -= (1 - lambda) * maxSimilarity
				}
			}
			if value > bestValue {
				best, bestValue = i, value
			}
		}
		selected = append(selected, remaining[best])
		remaining = append(remaining[:best], remaining[best+1:]...)
	}
	return selected
}

func deduplicateHits(hits []parsedHit, limit int) []parsedHit {
	seen := make(map[string]struct{}, len(hits))
	unique := make([]parsedHit, 0, len(hits))
	for _, hit := range hits {
		key := strings.Join(strings.Fields(strings.ToLower(hit.text)), " ")
		if key == "" {
			continue
		}
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		unique = append(unique, hit)
	}
	if limit >= 0 && len(unique) > limit {
		return unique[:limit]
	}
	return unique
}

func cosine(left, right []float32) float32 {
	if len(left) == 0 || len(left) != len(right) {
		return 0
	}
	var dot, leftNorm, rightNorm float32
	for i := range left {
		dot += left[i] * right[i]
		leftNorm += left[i] * left[i]
		rightNorm += right[i] * right[i]
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(leftNorm*rightNorm)))
}

// payloadString returns the first non-empty string value found across fields, in order.
func payloadString(payload map[string]*qdrant.Value, fields ...string) string {
	for _, field := range fields {
		if v, ok := payload[field]; ok {
			if sv := v.GetStringValue(); sv != "" {
				return sv
			}
		}
	}
	return ""
}

// payloadInt returns the integer payload value at field, if present.
func payloadInt(payload map[string]*qdrant.Value, field string) (int64, bool) {
	v, ok := payload[field]
	if !ok {
		return 0, false
	}
	if _, isInt := v.Kind.(*qdrant.Value_IntegerValue); !isInt {
		return 0, false
	}
	return v.GetIntegerValue(), true
}

// stitchNeighborChunks expands each hit that has file_path+chunk_index with the immediate
// previous/next chunk from the same file, fetched by a filter-only scroll (no vector search).
// This avoids handing the LLM a chunk truncated mid-function/mid-explanation. Best-effort:
// scroll failures are logged and simply leave the affected hits unstitched.
func (c *Client) stitchNeighborChunks(ctx context.Context, collection, repoID, component string, hits []parsedHit) {
	// file_path -> set of chunk_index already present in this result set (no need to refetch those).
	present := make(map[string]map[int64]bool)
	for _, h := range hits {
		if !h.hasChunkIndex || h.source == "" {
			continue
		}
		if present[h.source] == nil {
			present[h.source] = make(map[int64]bool)
		}
		present[h.source][h.chunkIndex] = true
	}
	if len(present) == 0 {
		return
	}

	// Fetch missing neighbors once per file_path (not once per hit).
	neighborText := make(map[string]map[int64]string)
	for filePath, indices := range present {
		var missing []int64
		for idx := range indices {
			for _, neighbor := range []int64{idx - 1, idx + 1} {
				if neighbor >= 0 && !indices[neighbor] {
					missing = append(missing, neighbor)
				}
			}
		}
		if len(missing) == 0 {
			continue
		}

		found, err := c.scrollByFilePath(ctx, collection, repoID, component, filePath, missing)
		if err != nil {
			c.log.Warn().Err(err).Str("collection", collection).Str("file_path", filePath).Msg("neighbor chunk stitch failed")
			continue
		}
		neighborText[filePath] = found
	}

	for i := range hits {
		h := &hits[i]
		if !h.hasChunkIndex || h.source == "" {
			continue
		}
		byIndex := neighborText[h.source]

		if prev, ok := lookupNeighbor(byIndex, present[h.source], h.chunkIndex-1); ok {
			h.text = prev + "\n\n" + h.text
		}
		if next, ok := lookupNeighbor(byIndex, present[h.source], h.chunkIndex+1); ok {
			h.text = h.text + "\n\n" + next
		}
	}
}

// lookupNeighbor returns the neighbor chunk's text if it was fetched via scroll (not already
// present in the original result set, which would make stitching it in a duplicate).
func lookupNeighbor(byIndex map[int64]string, present map[int64]bool, idx int64) (string, bool) {
	if idx < 0 || present[idx] {
		return "", false
	}
	text, ok := byIndex[idx]
	return text, ok
}

// scrollByFilePath fetches chunks for file_path at the given chunk_index values, no vector search.
func (c *Client) scrollByFilePath(ctx context.Context, collection, repoID, component, filePath string, chunkIndices []int64) (map[int64]string, error) {
	must := []*qdrant.Condition{qdrant.NewMatch("file_path", filePath)}
	if repoID != "" {
		must = append(must, qdrant.NewMatch("repo_id", repoID))
	}
	if component != "" {
		must = append(must, qdrant.NewMatch("component", component))
	}
	must = append(must, qdrant.NewMatchInts("chunk_index", chunkIndices...))

	resp, err := c.points.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: collection,
		Filter:         &qdrant.Filter{Must: must},
		Limit:          qdrant.PtrOf(uint32(len(chunkIndices))),
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		return nil, fmt.Errorf("scroll %s: %w", collection, err)
	}

	found := make(map[int64]string, len(resp.Result))
	for _, point := range resp.Result {
		if point.Payload == nil {
			continue
		}
		idx, ok := payloadInt(point.Payload, "chunk_index")
		if !ok {
			continue
		}
		text := payloadString(point.Payload, "chunk_text", "text")
		if text == "" {
			continue
		}
		found[idx] = text
	}
	return found, nil
}
