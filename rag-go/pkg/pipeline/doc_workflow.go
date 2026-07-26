package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type DocDecisionStatus string

const (
	StatusNoChangesRequired  DocDecisionStatus = "no_changes_required"
	StatusUpdateRequired     DocDecisionStatus = "update_required"
	StatusNewDocumentRequired DocDecisionStatus = "new_document_required"
)

type DocProfile struct {
	Kind             string
	RequiredSections []string
	Tone             string
	Audience         string
}

func defaultDocProfile() DocProfile {
	return DocProfile{
		Kind: "kb_article",
		RequiredSections: []string{
			"Summary",
			"Context",
			"Steps",
			"Validation",
			"Notes",
		},
		Tone:     "technical and concise",
		Audience: "support and engineering",
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
	DocRef         string  `json:"doc_ref"`
	Title          string  `json:"title"`
	WhyMatched     string  `json:"why_matched"`
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
	TargetDocRef   string   `json:"target_doc_ref"`
	PatchType      string   `json:"patch_type"`
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
	FactsCount       int `json:"facts_count"`
	UnknownsCount    int `json:"unknowns_count"`
	MatchedDocsCount int `json:"matched_docs_count"`
	MissingFactsCount int `json:"missing_facts_count"`
	ConflictsCount   int `json:"conflicts_count"`
	StaleFactsCount  int `json:"stale_facts_count"`
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

	if len(audit.ConflictingFacts) > 0 || len(audit.MissingFacts) > 0 {
		return DocDecision{Status: StatusUpdateRequired, Reason: "Relevant documentation exists but has gaps or conflicts with code evidence."}
	}

	return DocDecision{Status: StatusNoChangesRequired, Reason: "Relevant documentation is aligned with current code evidence."}
}

func (e *DefaultDocDecisionEngine) ComputeConfidence(extract DocExtractResult, audit DocAuditResult, decision DocDecision) float64 {
	base := 0.55

	if len(extract.Facts) > 0 {
		base += 0.10
	}
	if len(audit.MatchedDocs) > 0 {
		base += 0.10
	}
	if len(audit.ConflictingFacts) == 0 {
		base += 0.10
	}
	if len(audit.MissingFacts) == 0 {
		base += 0.10
	}
	if decision.Status == StatusNoChangesRequired {
		base += 0.05
	}

	if base > 0.99 {
		base = 0.99
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

	decision := p.engine.Decide(extract, audit)

	gen, err := p.runGenerate(ctx, req, decision, extractRaw, auditRaw, profile, docChunks, genDocChunks)
	if err != nil {
		return "", err
	}

	final := DocProcessOutput{
		Status:            decision.Status,
		ConfidenceOverall: p.engine.ComputeConfidence(extract, audit, decision),
		Topic:             chooseTopic(extract.Topic, req.QueryText),
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
		Warnings: uniqueStrings(gen.Warnings),
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
	raw, err := p.completeAndRepairJSON(ctx, "extract", schema, buildDocExtractPrompt(req, changeChunks, codeChunks))
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
	raw, err := p.completeAndRepairJSON(ctx, "audit", schema, buildDocAuditPrompt(req, extractRaw, docChunks, genDocChunks))
	if err != nil {
		return DocAuditResult{}, "", err
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
	raw, err := p.completeAndRepairJSON(ctx, "generate", schema, buildDocGeneratePrompt(req, decision, extractRaw, auditRaw, profile, docChunks, genDocChunks))
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

func (p *LLMDocProcessor) completeAndRepairJSON(ctx context.Context, stepName, schemaHint string, messages []Message) (string, error) {
	raw, err := p.llm.Complete(ctx, messages)
	if err != nil {
		return "", fmt.Errorf("vllm %s step: %w", stepName, err)
	}

	if json.Valid([]byte(raw)) {
		return raw, nil
	}

	repairRaw, repairErr := p.llm.Complete(ctx, buildDocRepairPrompt(stepName, schemaHint, raw))
	if repairErr != nil {
		return "", fmt.Errorf("invalid json in %s step and repair failed: %w", stepName, repairErr)
	}
	if !json.Valid([]byte(repairRaw)) {
		return "", fmt.Errorf("invalid json in %s step after repair", stepName)
	}

	return repairRaw, nil
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

func chooseTopic(candidate, fallback string) string {
	candidate = strings.TrimSpace(candidate)
	if candidate != "" {
		return candidate
	}
	return strings.TrimSpace(fallback)
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
