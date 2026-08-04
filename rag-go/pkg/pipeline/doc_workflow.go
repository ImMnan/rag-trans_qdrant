package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

type DocDecisionStatus string

const (
	StatusNoChangesRequired   DocDecisionStatus = "no_changes_required"
	StatusUpdateRequired      DocDecisionStatus = "update_required"
	StatusNewDocumentRequired DocDecisionStatus = "new_document_required"
)

type DocProfile struct {
	Kind             string
	RequiredSections []string
	Tone             string
	Audience         string
	StepStyle        string
	ExamplePolicy    string
	ValidationPolicy string
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
		Tone:             "technical and concise",
		Audience:         "support and engineering",
		StepStyle:        "numbered user actions with one concrete task per step",
		ExamplePolicy:    "include at least one runnable fenced example using the most relevant language such as bash, yaml, json, or go when the request is instructional, asks for configuration, commands, scripts, snippets, examples, setup, or how-to guidance",
		ValidationPolicy: "include a short validation subsection with the exact command or observable check the user should run after completing the steps whenever the evidence supports one",
	}
}

type DocDecision struct {
	Status DocDecisionStatus `json:"status"`
	Reason string            `json:"reason"`
}

type DocFact struct {
	Fact       string  `json:"fact"`
	Source     string  `json:"source"`
	Confidence float64 `json:"confidence"`
}

type DocExtractResult struct {
	Topic    string    `json:"topic"`
	Facts    []DocFact `json:"facts"`
	Unknowns []string  `json:"unknowns"`
}

type DocMatched struct {
	DocRef          string  `json:"doc_ref"`
	Title           string  `json:"title"`
	WhyMatched      string  `json:"why_matched"`
	MatchConfidence float64 `json:"match_confidence"`
}

type DocAuditResult struct {
	MatchedDocs      []DocMatched `json:"matched_docs"`
	MissingFacts     []string     `json:"missing_facts"`
	ConflictingFacts []string     `json:"conflicting_facts"`
	StaleFacts       []string     `json:"stale_facts"`
	Summary          string       `json:"summary"`
}

type DocDelta struct {
	TargetDocRef    string   `json:"target_doc_ref"`
	PatchType       string   `json:"patch_type"`
	ChangedSections []string `json:"changed_sections"`
	ChangesMarkdown string   `json:"changes_markdown"`
}

type GeneratedDocument struct {
	DocKind      string   `json:"doc_kind"`
	Title        string   `json:"title"`
	Summary      string   `json:"summary"`
	BodyMarkdown string   `json:"body_markdown"`
	Tags         []string `json:"tags"`
}

type DocGenerateResult struct {
	Delta    DocDelta          `json:"delta"`
	Document GeneratedDocument `json:"document"`
	Warnings []string          `json:"warnings"`
}

type EvidenceSummary struct {
	FactsCount        int `json:"facts_count"`
	UnknownsCount     int `json:"unknowns_count"`
	MatchedDocsCount  int `json:"matched_docs_count"`
	MissingFactsCount int `json:"missing_facts_count"`
	ConflictsCount    int `json:"conflicts_count"`
	StaleFactsCount   int `json:"stale_facts_count"`
}

type DocProcessOutput struct {
	Status            DocDecisionStatus  `json:"status"`
	ConfidenceOverall float64            `json:"confidence_overall"`
	Topic             string             `json:"topic"`
	MatchedDocs       []DocMatched       `json:"matched_docs"`
	DecisionReason    string             `json:"decision_reason"`
	EvidenceSummary   EvidenceSummary    `json:"evidence_summary"`
	Delta             *DocDelta          `json:"delta,omitempty"`
	Document          *GeneratedDocument `json:"document,omitempty"`
	Warnings          []string           `json:"warnings"`
}

type DocDecisionEngine interface {
	Decide(extract DocExtractResult, audit DocAuditResult) DocDecision
	ComputeConfidence(extract DocExtractResult, audit DocAuditResult, decision DocDecision) float64
}

type DefaultDocDecisionEngine struct {
	MinDocMatchConfidence float64
}

func NewDefaultDocDecisionEngine() *DefaultDocDecisionEngine {
	return &DefaultDocDecisionEngine{MinDocMatchConfidence: 0.55}
}

func (e *DefaultDocDecisionEngine) Decide(extract DocExtractResult, audit DocAuditResult) DocDecision {
	maxMatch := 0.0
	for _, d := range audit.MatchedDocs {
		if d.MatchConfidence > maxMatch {
			maxMatch = d.MatchConfidence
		}
	}

	if len(audit.MatchedDocs) == 0 || maxMatch < e.MinDocMatchConfidence {
		return DocDecision{Status: StatusNewDocumentRequired, Reason: "No sufficiently relevant existing documentation found."}
	}

	if len(audit.ConflictingFacts) > 0 || len(audit.MissingFacts) > 0 || len(audit.StaleFacts) > 0 {
		return DocDecision{Status: StatusUpdateRequired, Reason: "Relevant documentation exists but has gaps or conflicts with code evidence."}
	}

	if len(extract.Unknowns) > 0 {
		return DocDecision{Status: StatusUpdateRequired, Reason: "Relevant documentation exists, but unresolved unknowns prevent a no-change decision."}
	}

	if len(extract.Facts) == 0 {
		return DocDecision{Status: StatusUpdateRequired, Reason: "Relevant documentation exists, but there is not enough extracted evidence to confirm no changes are required."}
	}

	return DocDecision{Status: StatusNoChangesRequired, Reason: "Relevant documentation is aligned with current code evidence."}
}

func (e *DefaultDocDecisionEngine) ComputeConfidence(extract DocExtractResult, audit DocAuditResult, decision DocDecision) float64 {
	base := 0.40

	if len(extract.Facts) >= 2 {
		base += 0.10
	} else if len(extract.Facts) == 1 {
		base += 0.05
	}
	if len(extract.Unknowns) == 0 {
		base += 0.08
	}
	if len(audit.MatchedDocs) > 0 {
		base += 0.10
	}
	if len(audit.ConflictingFacts) == 0 {
		base += 0.08
	}
	if len(audit.MissingFacts) == 0 {
		base += 0.08
	}
	if len(audit.StaleFacts) == 0 {
		base += 0.08
	}
	if decision.Status == StatusNoChangesRequired && len(extract.Facts) > 0 && len(audit.MatchedDocs) > 0 {
		base += 0.03
	}

	if base > 0.95 {
		base = 0.95
	}
	if base < 0.0 {
		base = 0.0
	}

	return base
}

type DocProcessor interface {
	Process(ctx context.Context, req Request, changeChunks, codeChunks, docChunks, genDocChunks []string) (string, error)
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

func (p *LLMDocProcessor) Process(ctx context.Context, req Request, changeChunks, codeChunks, docChunks, genDocChunks []string) (string, error) {
	profile := defaultDocProfile()

	extract, extractRaw, err := p.runExtract(ctx, req, changeChunks, codeChunks)
	if err != nil {
		return "", err
	}

	audit, auditRaw, err := p.runAudit(ctx, req, extractRaw, docChunks, genDocChunks)
	if err != nil {
		return "", err
	}

	coverageWarnings := enforceAuditFactCoverage(req, extract, &audit, docChunks, genDocChunks)

	decision := p.engine.Decide(extract, audit)

	gen, err := p.runGenerate(ctx, req, decision, extractRaw, auditRaw, profile, docChunks, genDocChunks)
	if err != nil {
		return "", err
	}

	final := DocProcessOutput{
		Status:            decision.Status,
		ConfidenceOverall: p.engine.ComputeConfidence(extract, audit, decision),
		Topic:             strings.TrimSpace(req.QueryText),
		MatchedDocs:       audit.MatchedDocs,
		DecisionReason:    decision.Reason,
		EvidenceSummary: EvidenceSummary{
			FactsCount:        len(extract.Facts),
			UnknownsCount:     len(extract.Unknowns),
			MatchedDocsCount:  len(audit.MatchedDocs),
			MissingFactsCount: len(audit.MissingFacts),
			ConflictsCount:    len(audit.ConflictingFacts),
			StaleFactsCount:   len(audit.StaleFacts),
		},
		Warnings: uniqueStrings(append(gen.Warnings, coverageWarnings...)),
	}
	if topic := strings.TrimSpace(extract.Topic); topic != "" {
		final.Topic = topic
	}

	switch decision.Status {
	case StatusUpdateRequired:
		delta := gen.Delta
		if strings.TrimSpace(delta.TargetDocRef) == "" && len(audit.MatchedDocs) > 0 {
			delta.TargetDocRef = audit.MatchedDocs[0].DocRef
		}
		final.Delta = &delta
	case StatusNewDocumentRequired:
		doc := gen.Document
		if strings.TrimSpace(doc.DocKind) == "" {
			doc.DocKind = profile.Kind
		}
		final.Document = &doc
	}

	b, err := json.Marshal(final)
	if err != nil {
		return "", fmt.Errorf("marshal final doc output: %w", err)
	}
	return string(b), nil
}

func (p *LLMDocProcessor) runExtract(ctx context.Context, req Request, changeChunks, codeChunks []string) (DocExtractResult, string, error) {
	schema := `{"topic":"string","facts":[{"fact":"string","source":"change|code","confidence":0.0}],"unknowns":["string"]}`
	raw, err := p.completeAndRepairJSON(ctx, req, "extract", schema, buildDocExtractPrompt(req, changeChunks, codeChunks))
	if err != nil {
		return DocExtractResult{}, "", err
	}

	var out DocExtractResult
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return DocExtractResult{}, "", fmt.Errorf("parse extract result: %w", err)
	}
	normalizeExtract(&out)
	return out, raw, nil
}

func (p *LLMDocProcessor) runAudit(ctx context.Context, req Request, extractRaw string, docChunks, genDocChunks []string) (DocAuditResult, string, error) {
	schema := `{"matched_docs":[{"doc_ref":"string","title":"string","why_matched":"string","match_confidence":0.0}],"missing_facts":["string"],"conflicting_facts":["string"],"stale_facts":["string"],"summary":"string"}`
	raw, err := p.completeAndRepairJSON(ctx, req, "audit", schema, buildDocAuditPrompt(req, extractRaw, docChunks, genDocChunks))
	if err != nil {
		return DocAuditResult{}, "", err
	}

	if err := validateAuditJSONShape(raw); err != nil {
		return DocAuditResult{}, "", fmt.Errorf("invalid audit result shape: %w", err)
	}

	var out DocAuditResult
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return DocAuditResult{}, "", fmt.Errorf("parse audit result: %w", err)
	}
	normalizeAudit(&out)
	return out, raw, nil
}

func (p *LLMDocProcessor) runGenerate(
	ctx context.Context,
	req Request,
	decision DocDecision,
	extractRaw string,
	auditRaw string,
	profile DocProfile,
	docChunks []string,
	genDocChunks []string,
) (DocGenerateResult, error) {
	schema := `{"delta":{"target_doc_ref":"string","patch_type":"section_replace|add_section|remove_section|note_fix","changed_sections":["string"],"changes_markdown":"string"},"document":{"doc_kind":"kb_article","title":"string","summary":"string","body_markdown":"string","tags":["string"]},"warnings":["string"]}`
	raw, err := p.completeAndRepairJSON(ctx, req, "generate", schema, buildDocGeneratePrompt(req, decision, extractRaw, auditRaw, profile, docChunks, genDocChunks))
	if err != nil {
		return DocGenerateResult{}, err
	}

	var out DocGenerateResult
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		return DocGenerateResult{}, fmt.Errorf("parse generate result: %w", err)
	}
	normalizeGenerate(&out)
	return out, nil
}

func (p *LLMDocProcessor) completeAndRepairJSON(ctx context.Context, req Request, stepName, schemaHint string, messages []Message) (string, error) {
	maxTokens := ResolveDocStepTokenBudget(req, stepName, messages)
	raw, err := p.llm.Complete(ctx, messages, maxTokens)
	if err != nil {
		return "", fmt.Errorf("vllm %s step: %w", stepName, err)
	}

	if normalized, ok := normalizeJSONObject(raw); ok {
		return normalized, nil
	}

	repairPrompt := buildDocRepairPrompt(stepName, schemaHint, raw)
	repairMaxTokens := ResolveDocStepTokenBudget(req, "repair", repairPrompt)
	repairRaw, repairErr := p.llm.Complete(ctx, repairPrompt, repairMaxTokens)
	if repairErr != nil {
		return "", fmt.Errorf("invalid json in %s step and repair failed: %w", stepName, repairErr)
	}
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

	return "", false
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

func normalizeExtract(in *DocExtractResult) {
	if in.Facts == nil {
		in.Facts = []DocFact{}
	}
	if in.Unknowns == nil {
		in.Unknowns = []string{}
	}
	for i := range in.Facts {
		src := strings.ToLower(strings.TrimSpace(in.Facts[i].Source))
		if src != "change" && src != "code" {
			in.Facts[i].Source = "code"
		}
		if in.Facts[i].Confidence < 0.0 {
			in.Facts[i].Confidence = 0.0
		}
		if in.Facts[i].Confidence > 1.0 {
			in.Facts[i].Confidence = 1.0
		}
	}
}

func normalizeAudit(in *DocAuditResult) {
	if in.MatchedDocs == nil {
		in.MatchedDocs = []DocMatched{}
	}
	if in.MissingFacts == nil {
		in.MissingFacts = []string{}
	}
	if in.ConflictingFacts == nil {
		in.ConflictingFacts = []string{}
	}
	if in.StaleFacts == nil {
		in.StaleFacts = []string{}
	}
	for i := range in.MatchedDocs {
		if in.MatchedDocs[i].MatchConfidence < 0.0 {
			in.MatchedDocs[i].MatchConfidence = 0.0
		}
		if in.MatchedDocs[i].MatchConfidence > 1.0 {
			in.MatchedDocs[i].MatchConfidence = 1.0
		}
	}
}

func validateAuditJSONShape(raw string) error {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil {
		return err
	}

	required := []string{"matched_docs", "missing_facts", "conflicting_facts", "stale_facts", "summary"}
	missing := make([]string, 0)
	for _, key := range required {
		if _, ok := envelope[key]; !ok {
			missing = append(missing, key)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("missing required fields: %s", strings.Join(missing, ", "))
	}

	var matchedDocs []DocMatched
	if err := json.Unmarshal(envelope["matched_docs"], &matchedDocs); err != nil {
		return fmt.Errorf("matched_docs must be an array: %w", err)
	}

	for _, key := range []string{"missing_facts", "conflicting_facts", "stale_facts"} {
		var facts []string
		if err := json.Unmarshal(envelope[key], &facts); err != nil {
			return fmt.Errorf("%s must be an array of strings: %w", key, err)
		}
	}

	var summary string
	if err := json.Unmarshal(envelope["summary"], &summary); err != nil {
		return fmt.Errorf("summary must be a string: %w", err)
	}

	return nil
}

func enforceAuditFactCoverage(req Request, extract DocExtractResult, audit *DocAuditResult, docChunks, genDocChunks []string) []string {
	requiredFacts := parseRequestedFactsFromQuery(req.QueryText)
	if len(requiredFacts) == 0 {
		requiredFacts = make([]string, 0, len(extract.Facts))
		for _, fact := range extract.Facts {
			trimmed := strings.TrimSpace(fact.Fact)
			if trimmed == "" {
				continue
			}
			requiredFacts = append(requiredFacts, trimmed)
		}
	}
	if len(requiredFacts) == 0 || audit == nil || len(audit.MatchedDocs) == 0 {
		return nil
	}

	warnings := make([]string, 0)
	for _, fact := range requiredFacts {
		if factCoveredByMatchedDocs(fact, audit.MatchedDocs, docChunks, genDocChunks) {
			continue
		}
		audit.MissingFacts = appendUniqueCaseInsensitive(audit.MissingFacts, fact)
		warnings = append(warnings, "Requested fact is not present in matched documentation: "+fact)
	}

	return warnings
}

func parseRequestedFactsFromQuery(query string) []string {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil
	}

	lower := strings.ToLower(query)
	idx := strings.Index(lower, "facts to include")
	if idx == -1 {
		return nil
	}

	section := query[idx:]
	lines := strings.Split(section, "\n")
	facts := make([]string, 0)
	bulletPattern := regexp.MustCompile(`^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$`)
	for i, line := range lines {
		if i == 0 {
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if len(facts) > 0 {
				break
			}
			continue
		}
		if strings.HasPrefix(strings.ToLower(trimmed), "if relevant documentation exists") {
			break
		}
		if match := bulletPattern.FindStringSubmatch(trimmed); len(match) == 2 {
			fact := strings.TrimSpace(match[1])
			if fact != "" {
				facts = append(facts, fact)
			}
			continue
		}
		if len(facts) > 0 {
			break
		}
	}

	return facts
}

func factCoveredByMatchedDocs(fact string, matchedDocs []DocMatched, docChunks, genDocChunks []string) bool {
	fact = strings.TrimSpace(fact)
	if fact == "" {
		return true
	}

	factNorm := normalizeForMatch(fact)
	factTerms := significantTerms(factNorm)
	if len(factTerms) == 0 {
		return true
	}

	for _, doc := range matchedDocs {
		ref := strings.TrimSpace(doc.DocRef)
		if ref == "" {
			continue
		}

		bodies := collectSourceBodies(ref, docChunks)
		if len(bodies) == 0 {
			bodies = collectSourceBodies(ref, genDocChunks)
		}
		for _, body := range bodies {
			bodyNorm := normalizeForMatch(body)
			if bodyNorm == "" {
				continue
			}
			if strings.Contains(bodyNorm, factNorm) {
				return true
			}

			matched := 0
			for _, term := range factTerms {
				if strings.Contains(bodyNorm, term) {
					matched++
				}
			}
			ratio := float64(matched) / float64(len(factTerms))
			if matched >= 2 && ratio >= 0.60 {
				return true
			}
		}
	}

	return false
}

func collectSourceBodies(targetDocRef string, chunks []string) []string {
	targetDocRef = strings.TrimSpace(targetDocRef)
	if targetDocRef == "" || len(chunks) == 0 {
		return nil
	}

	seen := map[string]struct{}{}
	bodies := make([]string, 0)
	for _, chunk := range chunks {
		source, body, ok := parseSourceChunk(chunk)
		if !ok || !sourceMatches(targetDocRef, source) {
			continue
		}
		body = strings.TrimSpace(body)
		if body == "" {
			continue
		}
		if _, exists := seen[body]; exists {
			continue
		}
		seen[body] = struct{}{}
		bodies = append(bodies, body)
	}

	return bodies
}

func parseSourceChunk(chunk string) (source, body string, ok bool) {
	chunk = strings.TrimSpace(chunk)
	if chunk == "" {
		return "", "", false
	}

	firstLine, rest, hasNewline := strings.Cut(chunk, "\n")
	if !hasNewline {
		return "", "", false
	}

	firstLine = strings.TrimSpace(firstLine)
	if !strings.HasPrefix(firstLine, "[source:") || !strings.HasSuffix(firstLine, "]") {
		return "", "", false
	}

	source = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(firstLine, "[source:"), "]"))
	if source == "" {
		return "", "", false
	}

	return source, rest, true
}

func sourceMatches(targetDocRef, source string) bool {
	target := strings.TrimSpace(targetDocRef)
	current := strings.TrimSpace(source)
	if target == "" || current == "" {
		return false
	}

	if strings.EqualFold(target, current) {
		return true
	}

	targetLower := strings.ToLower(target)
	currentLower := strings.ToLower(current)
	return strings.HasSuffix(currentLower, "/"+targetLower) || strings.HasSuffix(targetLower, "/"+currentLower)
}

func normalizeForMatch(text string) string {
	text = strings.ToLower(strings.TrimSpace(text))
	if text == "" {
		return ""
	}
	re := regexp.MustCompile(`[^a-z0-9]+`)
	text = re.ReplaceAllString(text, " ")
	return strings.Join(strings.Fields(text), " ")
}

func significantTerms(text string) []string {
	if text == "" {
		return nil
	}
	stop := map[string]struct{}{
		"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {}, "by": {}, "for": {},
		"from": {}, "in": {}, "is": {}, "it": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
		"to": {}, "using": {}, "with": {}, "this": {}, "these": {}, "those": {}, "ability": {}, "support": {},
	}

	raw := strings.Fields(text)
	terms := make([]string, 0, len(raw))
	seen := map[string]struct{}{}
	for _, token := range raw {
		if len(token) < 4 {
			continue
		}
		if _, isStop := stop[token]; isStop {
			continue
		}
		if _, exists := seen[token]; exists {
			continue
		}
		seen[token] = struct{}{}
		terms = append(terms, token)
	}
	return terms
}

func appendUniqueCaseInsensitive(values []string, candidate string) []string {
	candidate = strings.TrimSpace(candidate)
	if candidate == "" {
		return values
	}
	for _, existing := range values {
		if strings.EqualFold(strings.TrimSpace(existing), candidate) {
			return values
		}
	}
	return append(values, candidate)
}

func normalizeGenerate(in *DocGenerateResult) {
	if in.Warnings == nil {
		in.Warnings = []string{}
	}
	if in.Delta.ChangedSections == nil {
		in.Delta.ChangedSections = []string{}
	}
	if in.Document.Tags == nil {
		in.Document.Tags = []string{}
	}
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
