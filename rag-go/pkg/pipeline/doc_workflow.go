package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type DocDecisionStatus string

const (
	StatusNoChangesRequired   DocDecisionStatus = "no_changes_required"
	StatusUpdateRequired      DocDecisionStatus = "update_required"
	StatusNewDocumentRequired DocDecisionStatus = "new_document_required"
)

// Coverage is how much of the query the existing documentation already answers.
const (
	CoverageComplete = "complete"
	CoveragePartial  = "partial"
	CoverageNone     = "none"
)

// minChunkRelevance is the floor for keeping a documentation chunk after triage.
const minChunkRelevance = 0.5

type DocProfile struct {
	Kind             string
	RequiredSections []string
	Tone             string
	Audience         string
}

func defaultDocProfile() DocProfile {
	return DocProfile{
		Kind: "kb_article", //for now I am keeping this hardcoded.
		RequiredSections: []string{
			"Summary",
			"Context",
			"Steps",
			"Validation",
			"Notes",
		},
		Tone:     "technical and concise",
		Audience: "users and operators of this product",
	}
}

type DocDecision struct {
	Status DocDecisionStatus `json:"status"`
	Reason string            `json:"reason"`
}

// DocChunkVerdict is triage's judgement on one indexed documentation chunk.
type DocChunkVerdict struct {
	Index     int     `json:"index"`
	Relevance float64 `json:"relevance"`
	Why       string  `json:"why"`
}

// DocMatched is a documentation chunk that survived triage, reported for attribution.
type DocMatched struct {
	DocRef     string  `json:"doc_ref"`
	ChunkIndex int     `json:"chunk_index"`
	Relevance  float64 `json:"relevance"`
	WhyMatched string  `json:"why_matched"`
}

type DocTriageResult struct {
	Topic          string            `json:"topic"`
	Coverage       string            `json:"coverage"`
	RelevantChunks []DocChunkVerdict `json:"relevant_chunks"`
	MissingPoints  []string          `json:"missing_points"`
	Reason         string            `json:"reason"`
}

// DocComposeResult is one shape for both branches: whether the answer was built on an
// existing document or written fresh, it is still a title plus a body.
type DocComposeResult struct {
	Title        string   `json:"title"`
	BodyMarkdown string   `json:"body_markdown"`
	Corrections  []string `json:"corrections"`
	Warnings     []string `json:"warnings"`
}

type DocProcessOutput struct {
	Status            DocDecisionStatus `json:"status"`
	Coverage          string            `json:"coverage"`
	ConfidenceOverall float64           `json:"confidence_overall"`
	Topic             string            `json:"topic"`
	Title             string            `json:"title"`
	BodyMarkdown      string            `json:"body_markdown"`
	Corrections       []string          `json:"corrections"`
	MissingPoints     []string          `json:"missing_points"`
	MatchedDocs       []DocMatched      `json:"matched_docs"`
	DecisionReason    string            `json:"decision_reason"`
	Warnings          []string          `json:"warnings"`
}

type DocDecisionEngine interface {
	Decide(triage DocTriageResult, compose DocComposeResult) DocDecision
	ComputeConfidence(triage DocTriageResult, compose DocComposeResult) float64
}

type DefaultDocDecisionEngine struct{}

func NewDefaultDocDecisionEngine() *DefaultDocDecisionEngine {
	return &DefaultDocDecisionEngine{}
}

// Decide reports what should happen to the documentation. Corrections come from verifying
// against source code, so any correction means the doc is outdated no matter how complete
// triage judged it to be.
func (e *DefaultDocDecisionEngine) Decide(triage DocTriageResult, compose DocComposeResult) DocDecision {
	if triage.Coverage == CoverageNone {
		return DocDecision{Status: StatusNewDocumentRequired, Reason: "No existing documentation covers this query; the answer was written from source evidence."}
	}

	if len(compose.Corrections) > 0 {
		return DocDecision{Status: StatusUpdateRequired, Reason: "Source code contradicts the existing documentation; the printed steps were corrected."}
	}

	if triage.Coverage == CoveragePartial {
		return DocDecision{Status: StatusUpdateRequired, Reason: "Existing documentation answers only part of the query; the remainder came from source evidence."}
	}

	return DocDecision{Status: StatusNoChangesRequired, Reason: "Existing documentation answers the query and matches the current source code."}
}

func (e *DefaultDocDecisionEngine) ComputeConfidence(triage DocTriageResult, compose DocComposeResult) float64 {
	base := 0.40

	switch triage.Coverage {
	case CoverageComplete:
		base += 0.25
	case CoveragePartial:
		base += 0.10
	}

	strong := 0
	for _, c := range triage.RelevantChunks {
		if c.Relevance >= 0.75 {
			strong++
		}
	}
	if strong >= 2 {
		base += 0.15
	} else if strong == 1 {
		base += 0.08
	}

	if len(compose.Corrections) == 0 {
		base += 0.10
	}
	if len(compose.Warnings) == 0 {
		base += 0.10
	}
	if len(triage.MissingPoints) > 2 {
		base -= 0.10
	}

	return clampFloat(base, 0.0, 0.95)
}

func clampFloat(v, minV, maxV float64) float64 {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

type DocProcessor interface {
	Process(ctx context.Context, req Request, codeChunks, docChunks []string) (string, error)
}

type LLMDocProcessor struct {
	llm    VLLMCompleter
	engine DocDecisionEngine
}

func NewLLMDocProcessor(llm VLLMCompleter, engine DocDecisionEngine) *LLMDocProcessor {
	if engine == nil {
		engine = NewDefaultDocDecisionEngine()
	}
	return &LLMDocProcessor{llm: llm, engine: engine}
}

// Process triages the retrieved documentation, then composes the answer. Source evidence is
// always given to the compose step: triage can tell whether a document is incomplete, but
// only the code can tell whether it is stale.
func (p *LLMDocProcessor) Process(ctx context.Context, req Request, codeChunks, docChunks []string) (string, error) {
	profile := defaultDocProfile()
	indexed := p.fitTriageChunks(req, indexDocChunks(docChunks))

	triage, triageWarnings, err := p.runTriage(ctx, req, indexed)
	if err != nil {
		return "", err
	}

	kept := selectRelevantChunks(indexed, triage.RelevantChunks)
	if len(kept.Chunks) == 0 {
		triage.Coverage = CoverageNone
	}

	compose, err := p.runCompose(ctx, req, profile, triage, kept.Chunks, codeChunks)
	if err != nil {
		return "", err
	}

	decision := p.engine.Decide(triage, compose)

	final := DocProcessOutput{
		Status:            decision.Status,
		Coverage:          triage.Coverage,
		ConfidenceOverall: p.engine.ComputeConfidence(triage, compose),
		Topic:             strings.TrimSpace(req.QueryText),
		Title:             compose.Title,
		BodyMarkdown:      compose.BodyMarkdown,
		Corrections:       compose.Corrections,
		MissingPoints:     triage.MissingPoints,
		MatchedDocs:       kept.Matched,
		DecisionReason:    decision.Reason,
		Warnings:          uniqueStrings(append(append(compose.Warnings, triageWarnings...), kept.Warnings...)),
	}
	if topic := strings.TrimSpace(triage.Topic); topic != "" {
		final.Topic = topic
	}

	b, err := json.Marshal(final)
	if err != nil {
		return "", fmt.Errorf("marshal final doc output: %w", err)
	}
	return string(b), nil
}

// runTriage asks which documentation chunks actually answer the query. With no documentation
// retrieved there is nothing to judge, so the call is skipped entirely.
func (p *LLMDocProcessor) runTriage(ctx context.Context, req Request, indexed []string) (DocTriageResult, []string, error) {
	if len(indexed) == 0 {
		return DocTriageResult{
			Coverage:       CoverageNone,
			RelevantChunks: []DocChunkVerdict{},
			MissingPoints:  []string{},
			Reason:         "No documentation chunks were retrieved for this query.",
		}, nil, nil
	}

	schema := `{"topic":"<topic>","coverage":"complete|partial|none","relevant_chunks":[{"index":0,"relevance":0.0,"why":"<why>"}],"missing_points":["<what the docs do not answer>"],"reason":"<reason>"}`
	req.ReportProgress("messages_assembled", "Assembled triage messages, now awaiting LLM response...", 65)
	raw, err := p.completeAndRepairJSON(ctx, req, "triage", schema, buildDocTriagePrompt(req, indexed))
	if err != nil {
		return DocTriageResult{}, nil, err
	}

	var out DocTriageResult
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return DocTriageResult{}, nil, fmt.Errorf("parse triage result: %w", err)
	}

	warnings := normalizeTriage(&out, len(indexed))
	return out, warnings, nil
}

// runCompose asks for a labelled plain-text envelope rather than JSON. The body carries
// fenced code, quotes, and pasted log lines, and making the model escape all of that inside
// a JSON string is the single biggest source of unparseable output.
func (p *LLMDocProcessor) runCompose(
	ctx context.Context,
	req Request,
	profile DocProfile,
	triage DocTriageResult,
	keptDocChunks []string,
	codeChunks []string,
) (DocComposeResult, error) {
	messages := p.fitComposePrompt(req, profile, triage, keptDocChunks, codeChunks)

	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		if attempt > 0 {
			messages = append(messages, Message{
				Role: "user",
				Content: "Your previous answer was rejected: " + lastErr.Error() + ". " +
					"Answer again in the same label format: a TITLE: line, then optional CORRECTION: and WARNING: lines, " +
					"then a BODY: line followed by the complete markdown document. Do not emit JSON and do not use placeholders.",
			})
		}

		req.ReportProgress("messages_assembled", "Assembled document-generation messages, now awaiting LLM response...", 80)
		raw, err := p.llm.Complete(ctx, messages, ResolveDocStepTokenBudget(req, "compose", messages))
		if err != nil {
			return DocComposeResult{}, fmt.Errorf("vllm compose step: %w", err)
		}
		req.ReportProgress("document_generation_complete", "Document generation complete", 95)

		out := parseComposeEnvelope(raw)
		if lastErr = validateComposeResult(out); lastErr == nil {
			return out, nil
		}
	}

	return DocComposeResult{}, fmt.Errorf("compose step produced unusable content: %w", lastErr)
}

// parseComposeEnvelope reads the TITLE/CORRECTION/WARNING/BODY labels. Everything after the
// BODY: line is taken verbatim, so the document may contain any character.
func parseComposeEnvelope(raw string) DocComposeResult {
	out := DocComposeResult{Corrections: []string{}, Warnings: []string{}}

	lines := strings.Split(strings.ReplaceAll(strings.TrimSpace(raw), "\r\n", "\n"), "\n")
	body := make([]string, 0, len(lines))
	inBody := false

	for _, line := range lines {
		if inBody {
			body = append(body, line)
			continue
		}

		trimmed := strings.TrimSpace(line)
		switch {
		case hasLabel(trimmed, "BODY:"):
			inBody = true
			if rest := strings.TrimSpace(trimmed[len("BODY:"):]); rest != "" {
				body = append(body, rest)
			}
		case hasLabel(trimmed, "TITLE:"):
			out.Title = strings.TrimSpace(trimmed[len("TITLE:"):])
		case hasLabel(trimmed, "CORRECTION:"):
			if v := strings.TrimSpace(trimmed[len("CORRECTION:"):]); v != "" && !isPlaceholderContent(v) {
				out.Corrections = append(out.Corrections, v)
			}
		case hasLabel(trimmed, "WARNING:"):
			if v := strings.TrimSpace(trimmed[len("WARNING:"):]); v != "" && !isPlaceholderContent(v) {
				out.Warnings = append(out.Warnings, v)
			}
		case trimmed == "":
			// Blank lines before BODY: are formatting, not content.
		default:
			// Unlabelled prose before BODY: is the model narrating; treat it as the document.
			inBody = true
			body = append(body, line)
		}
	}

	out.BodyMarkdown = normalizeBodyMarkdown(strings.Join(body, "\n"))
	if isPlaceholderContent(out.Title) {
		out.Title = deriveTitle(out.BodyMarkdown)
	}
	return out
}

// normalizeBodyMarkdown removes an accidental outer fence around the complete
// document so inner code fences remain Markdown fences instead of plain text.
func normalizeBodyMarkdown(body string) string {
	trimmed := strings.TrimSpace(strings.ReplaceAll(body, "\r\n", "\n"))
	lines := strings.Split(trimmed, "\n")
	if len(lines) < 3 {
		return trimmed
	}

	first := strings.TrimSpace(lines[0])
	last := strings.TrimSpace(lines[len(lines)-1])
	if last != "```" {
		return trimmed
	}

	openingLine := 0
	if !isMarkdownFence(first) {
		if len(lines) < 4 {
			return trimmed
		}
		found := false
		for i := 0; i < len(lines)-2; i++ {
			if strings.EqualFold(strings.TrimSpace(lines[i]), "BODY:") && isMarkdownFence(strings.TrimSpace(lines[i+1])) {
				openingLine = i
				found = true
				break
			}
		}
		if !found {
			return trimmed
		}
		content := append([]string{}, lines[:openingLine]...)
		content = append(content, lines[openingLine+2:len(lines)-1]...)
		return strings.TrimSpace(strings.Join(content, "\n"))
	}

	return strings.TrimSpace(strings.Join(lines[openingLine+1:len(lines)-1], "\n"))
}

func isMarkdownFence(line string) bool {
	if !strings.HasPrefix(line, "```") {
		return false
	}
	language := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(line, "```")))
	return language == "markdown" || language == "md"
}

func hasLabel(line, label string) bool {
	return len(line) >= len(label) && strings.EqualFold(line[:len(label)], label)
}

// deriveTitle falls back to the document's first heading or first line.
func deriveTitle(body string) string {
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "```") {
			continue
		}
		return strings.TrimSpace(strings.TrimLeft(trimmed, "#* "))
	}
	return ""
}

// fitTriageChunks drops trailing documentation chunks until the triage call has room for its
// verdicts. Trimming the tail keeps the remaining [chunk N] labels contiguous from zero, so
// the indices triage returns still address this slice.
func (p *LLMDocProcessor) fitTriageChunks(req Request, indexed []string) []string {
	if req.TokenLimit > 0 {
		return indexed
	}

	for len(indexed) > 1 && ResolveDocStepTokenBudget(req, "triage", buildDocTriagePrompt(req, indexed)) < minTriageOutputTokens {
		indexed = indexed[:len(indexed)-1]
	}
	return indexed
}

// fitComposePrompt drops trailing context until the compose call has at least
// minGenerateOutputTokens of room, shedding kept docs before code.
func (p *LLMDocProcessor) fitComposePrompt(
	req Request,
	profile DocProfile,
	triage DocTriageResult,
	keptDocChunks []string,
	codeChunks []string,
) []Message {
	build := func() []Message {
		return buildDocComposePrompt(req, profile, triage, keptDocChunks, codeChunks)
	}

	messages := build()
	if req.TokenLimit > 0 {
		return messages
	}

	for ResolveDocStepTokenBudget(req, "compose", messages) < minGenerateOutputTokens {
		switch {
		case len(keptDocChunks) > 1:
			keptDocChunks = keptDocChunks[:len(keptDocChunks)-1]
		case len(codeChunks) > 0:
			codeChunks = codeChunks[:len(codeChunks)-1]
		default:
			return messages
		}
		messages = build()
	}

	return messages
}

// indexDocChunks labels each documentation chunk with the index triage refers to it by.
// An index cannot be hallucinated the way a filename can.
func indexDocChunks(docChunks []string) []string {
	indexed := make([]string, 0, len(docChunks))
	for _, c := range docChunks {
		if strings.TrimSpace(c) == "" {
			continue
		}
		indexed = append(indexed, fmt.Sprintf("[chunk %d]\n%s", len(indexed), c))
	}
	return indexed
}

// triageSelection is what survived triage, plus a record of what did not.
type triageSelection struct {
	Chunks   []string
	Matched  []DocMatched
	Warnings []string
}

func selectRelevantChunks(indexed []string, verdicts []DocChunkVerdict) triageSelection {
	sel := triageSelection{
		Chunks:   make([]string, 0, len(verdicts)),
		Matched:  make([]DocMatched, 0, len(verdicts)),
		Warnings: make([]string, 0),
	}

	seen := make(map[int]struct{}, len(verdicts))
	for _, v := range verdicts {
		if v.Index < 0 || v.Index >= len(indexed) {
			continue
		}
		if _, dup := seen[v.Index]; dup {
			continue
		}
		seen[v.Index] = struct{}{}

		ref := chunkSource(indexed[v.Index])
		if v.Relevance < minChunkRelevance {
			sel.Warnings = append(sel.Warnings, fmt.Sprintf(
				"Dropped documentation %s (chunk %d, relevance %.2f); it was not close enough to the query to quote.",
				ref, v.Index, v.Relevance))
			continue
		}

		sel.Chunks = append(sel.Chunks, indexed[v.Index])
		sel.Matched = append(sel.Matched, DocMatched{
			DocRef:     ref,
			ChunkIndex: v.Index,
			Relevance:  v.Relevance,
			WhyMatched: strings.TrimSpace(v.Why),
		})
	}

	if ignored := len(indexed) - len(seen); ignored > 0 {
		sel.Warnings = append(sel.Warnings, fmt.Sprintf(
			"Triage judged %d of %d retrieved documentation chunks irrelevant to this query.", ignored, len(indexed)))
	}

	return sel
}

// chunkSource reads the [source: ...] label off an indexed chunk.
func chunkSource(chunk string) string {
	for _, line := range strings.Split(chunk, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[source:") && strings.HasSuffix(line, "]") {
			if ref := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "[source:"), "]")); ref != "" {
				return ref
			}
		}
	}
	return "unattributed chunk"
}

func normalizeTriage(in *DocTriageResult, chunkCount int) []string {
	warnings := make([]string, 0)

	if in.RelevantChunks == nil {
		in.RelevantChunks = []DocChunkVerdict{}
	}
	if in.MissingPoints == nil {
		in.MissingPoints = []string{}
	}

	valid := make([]DocChunkVerdict, 0, len(in.RelevantChunks))
	for _, c := range in.RelevantChunks {
		if c.Index < 0 || c.Index >= chunkCount {
			warnings = append(warnings, fmt.Sprintf("Triage referenced documentation chunk %d, which was not retrieved; ignoring it.", c.Index))
			continue
		}
		c.Relevance = clampFloat(c.Relevance, 0.0, 1.0)
		valid = append(valid, c)
	}
	in.RelevantChunks = valid

	switch strings.ToLower(strings.TrimSpace(in.Coverage)) {
	case CoverageComplete:
		in.Coverage = CoverageComplete
	case CoveragePartial:
		in.Coverage = CoveragePartial
	default:
		in.Coverage = CoverageNone
	}

	// Claiming full coverage while listing gaps is a contradiction; trust the gaps.
	if in.Coverage == CoverageComplete && len(in.MissingPoints) > 0 {
		in.Coverage = CoveragePartial
	}

	return warnings
}

func validateComposeResult(out DocComposeResult) error {
	if isPlaceholderContent(out.BodyMarkdown) {
		return fmt.Errorf("body_markdown is empty or placeholder")
	}
	if isPlaceholderContent(out.Title) {
		return fmt.Errorf("title is empty or placeholder")
	}
	return nil
}

func isPlaceholderContent(value string) bool {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return true
	}
	if strings.HasPrefix(trimmed, "<") && strings.HasSuffix(trimmed, ">") {
		return true
	}
	switch strings.ToLower(trimmed) {
	case "string", "null", "n/a", "kb_article":
		return true
	}
	return false
}

func (p *LLMDocProcessor) completeAndRepairJSON(ctx context.Context, req Request, stepName, schemaHint string, messages []Message) (string, error) {
	maxTokens := ResolveDocStepTokenBudget(req, stepName, messages)
	raw, err := p.llm.Complete(ctx, messages, maxTokens)
	if err != nil {
		return "", fmt.Errorf("vllm %s step: %w", stepName, err)
	}
	req.ReportProgress("triage_complete", "Documentation triage complete", 70)

	if normalized, ok := normalizeJSONObject(raw); ok {
		return normalized, nil
	}

	repairPrompt := buildDocRepairPrompt(stepName, schemaHint, raw)
	repairMaxTokens := ResolveDocStepTokenBudget(req, "repair", repairPrompt)
	repairRaw, repairErr := p.llm.Complete(ctx, repairPrompt, repairMaxTokens)
	if repairErr != nil {
		return "", fmt.Errorf("invalid json in %s step and repair failed: %w", stepName, repairErr)
	}
	req.ReportProgress("triage_repair_complete", "Documentation triage repair complete", 75)
	normalizedRepair, ok := normalizeJSONObject(repairRaw)
	if !ok {
		return "", fmt.Errorf("invalid json in %s step after repair", stepName)
	}

	return normalizedRepair, nil
}

func normalizeJSONObject(raw string) (string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false
	}
	if json.Valid([]byte(raw)) {
		return raw, true
	}

	cleaned := strings.TrimSpace(stripCodeFence(raw))
	if cleaned != raw && json.Valid([]byte(cleaned)) {
		return cleaned, true
	}

	for _, candidate := range []string{cleaned, raw} {
		if extracted, ok := extractBalancedJSON(candidate); ok {
			return extracted, true
		}
	}

	// Markdown bodies routinely arrive with real line breaks and trailing commas; fix those
	// mechanically rather than spending a repair round-trip on them.
	for _, candidate := range []string{cleaned, raw} {
		repaired := sanitizeJSONText(candidate)
		if json.Valid([]byte(repaired)) {
			return repaired, true
		}
		if extracted, ok := extractBalancedJSON(repaired); ok {
			return extracted, true
		}
	}

	return "", false
}

// sanitizeJSONText escapes control characters that appear literally inside string values
// and drops trailing commas, without touching anything outside a string.
func sanitizeJSONText(raw string) string {
	var b strings.Builder
	b.Grow(len(raw))

	inString := false
	escaped := false
	for i := 0; i < len(raw); i++ {
		ch := raw[i]

		if inString {
			switch {
			case escaped:
				escaped = false
				b.WriteByte(ch)
			case ch == '\\':
				escaped = true
				b.WriteByte(ch)
			case ch == '"':
				inString = false
				b.WriteByte(ch)
			case ch == '\n':
				b.WriteString(`\n`)
			case ch == '\r':
				b.WriteString(`\r`)
			case ch == '\t':
				b.WriteString(`\t`)
			case ch < 0x20:
				fmt.Fprintf(&b, `\u%04x`, ch)
			default:
				b.WriteByte(ch)
			}
			continue
		}

		if ch == '"' {
			inString = true
			b.WriteByte(ch)
			continue
		}
		if ch == ',' && nextNonSpaceIsCloser(raw, i+1) {
			continue
		}
		b.WriteByte(ch)
	}

	return b.String()
}

func nextNonSpaceIsCloser(raw string, from int) bool {
	for i := from; i < len(raw); i++ {
		switch raw[i] {
		case ' ', '\t', '\n', '\r':
			continue
		case '}', ']':
			return true
		default:
			return false
		}
	}
	return false
}

func stripCodeFence(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if !strings.HasPrefix(trimmed, "```") {
		return raw
	}

	lines := strings.Split(trimmed, "\n")
	if len(lines) < 2 {
		return raw
	}
	if !strings.HasPrefix(lines[0], "```") {
		return raw
	}

	end := len(lines)
	if strings.TrimSpace(lines[len(lines)-1]) == "```" {
		end--
	}
	if end <= 1 {
		return raw
	}

	return strings.Join(lines[1:end], "\n")
}

func extractBalancedJSON(raw string) (string, bool) {
	start := -1
	for i, r := range raw {
		if r == '{' || r == '[' {
			start = i
			break
		}
	}
	if start == -1 {
		return "", false
	}

	depth := 0
	inString := false
	escaped := false
	for i := start; i < len(raw); i++ {
		ch := raw[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}

		switch ch {
		case '"':
			inString = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				candidate := strings.TrimSpace(raw[start : i+1])
				if json.Valid([]byte(candidate)) {
					return candidate, true
				}
			}
		}
	}

	return "", false
}

func uniqueStrings(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, v := range values {
		s := strings.TrimSpace(v)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}
