package pipeline

import (
	"fmt"
	"strings"
	"time"
)

// buildPrompt assembles the LLM messages from retrieved chunks.
func buildPrompt(req Request, changeChunks, codeChunks []string) []Message {
	changeCtx := joinChunks(changeChunks, "No change data found.")
	codeCtx := joinChunks(codeChunks, "No source context found.")

	if strings.EqualFold(strings.TrimSpace(req.Type), "standard") {
		today := time.Now().Format("2006-01-02")

		systemPrompt := "You are a senior engineer producing product release summaries. " +
			"Use only the provided context. Structure your answer with these sections:\n" +
			"0. **Time Window Interpreted** - print the exact date boundary you applied in YYYY-MM-DD format (for example: Window: 2026-08-02 to 2026-08-12), using ONLY the provided Today date as reference for relative periods.\n" +
			"1. **What Changed** - describe the commits/diffs concisely.\n" +
			"2. **User Impact** - explain what end-users will notice or need to act on.\n" +
			"3. **Security & Performance** - flag any security fixes or performance optimizations; " +
			"write 'None identified' if absent.\n" +
			"4. **Time Scope** - if the request specifies a timeframe (for example: last 10 days, between April and May), include only changes explicitly tied to that timeframe and treat Today as the reference point. Exclude anything outside it. Never infer the window from commit history or repository dates. Never infer or substitute a fallback timeframe. If there are no clearly in-range dated changes, state 'No changes found in the requested timeframe based on provided context.' and do not list out-of-range items.\n"

		userPrompt := fmt.Sprintf("## Today\n%s\n\n## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Request\n%s",
			today, changeCtx, codeCtx, req.QueryText)

		return []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		}
	}

	directPrompt := fmt.Sprintf(
		"Answer the user question using only the context below. "+
			"Be concise and factual. If asked whether a feature is supported, answer with 'Yes' or 'No' "+
			"and include when it first appears in the provided context if available; "+
			"otherwise say 'Unknown based on provided context'.\n\n"+
			"## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Question\n%s",
		changeCtx, codeCtx, req.QueryText,
	)

	return []Message{{Role: "user", Content: directPrompt}}
}

func joinChunks(chunks []string, fallback string) string {
	if len(chunks) == 0 {
		return fallback
	}
	return strings.Join(chunks, "\n---\n")
}

func buildDocExtractPrompt(req Request, changeChunks, codeChunks []string) []Message {
	changeCtx := joinChunks(changeChunks, "No change data found.")
	codeCtx := joinChunks(codeChunks, "No source context found.")

	systemPrompt := "You extract product and implementation facts from code evidence. " +
		"Use only the provided change/code context. " +
		"Return strict JSON and do not add markdown fences."

	userPrompt := fmt.Sprintf(
		"Return JSON with this shape only:\n"+
			"{\"topic\":string,\"facts\":[{\"fact\":string,\"source\":\"change|code\",\"confidence\":number}],\"unknowns\":[string]}\n\n"+
			"Rules:\n"+
			"- Extract factual statements only from evidence.\n"+
			"- If uncertain, list the point under unknowns.\n"+
			"- confidence range must be 0.0 to 1.0.\n\n"+
			"## Original Query\n%s\n\n"+
			"## Original Answer Type\n%s\n\n"+
			"## Diff / Change Hunks\n%s\n\n"+
			"## Source / Code Reference\n%s",
		req.QueryText, req.Type, changeCtx, codeCtx,
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

func buildDocAuditPrompt(req Request, extractedFactsJSON string, docChunks, genDocChunks []string) []Message {
	docCtx := joinChunks(docChunks, "No documentation context found.")
	genDocCtx := joinChunks(genDocChunks, "No generated documentation context found.")

	systemPrompt := "You audit existing documentation against extracted code facts. " +
		"Use only the provided facts and doc context. " +
		"Return strict JSON and do not add markdown fences."

	userPrompt := fmt.Sprintf(
		"Return JSON with this shape only:\n"+
			"{\"matched_docs\":[{\"doc_ref\":string,\"title\":string,\"why_matched\":string,\"match_confidence\":number}],\"missing_facts\":[string],\"conflicting_facts\":[string],\"stale_facts\":[string],\"summary\":string}\n\n"+
			"Rules:\n"+
			"- A matched doc must directly relate to the query topic.\n"+
			"- Treat code-derived facts as ground truth.\n"+
			"- match_confidence range must be 0.0 to 1.0.\n"+
			"- Each documentation chunk is prefixed with \"[source: <filename>]\" on its own line. "+
			"When a chunk matches the query topic, copy that exact filename (e.g. \"docs/gatling.md\") into the doc_ref field. "+
			"Never leave doc_ref empty when a [source: ...] label is present in the matched chunk.\n\n"+
			"## Original Query\n%s\n\n"+
			"## Extracted Facts JSON\n%s\n\n"+
			"## Existing Documentation Chunks\n%s\n\n"+
			"## Existing Generated Documentation Chunks\n%s",
		req.QueryText, extractedFactsJSON, docCtx, genDocCtx,
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

func buildDocGeneratePrompt(
	req Request,
	decision DocDecision,
	extractedFactsJSON string,
	auditJSON string,
	profile DocProfile,
	docChunks []string,
	genDocChunks []string,
) []Message {
	docCtx := joinChunks(docChunks, "No documentation context found.")
	genDocCtx := joinChunks(genDocChunks, "No generated documentation context found.")

	codeBlockRule := "In Steps and Validation sections, each actionable step must include at least one triple-backtick fenced code block with the correct language tag (bash, yaml, json, go, etc). The Validation section must end with a fenced bash block containing the exact verification command(s)."
	if isInstructionalRequest(req.QueryText) {
		codeBlockRule = "For every numbered step that involves a file, option, flag, or command: " +
			"write the step description, then on the very next line output a triple-backtick fenced code block " +
			"using the correct language tag (yaml, bash, json, go, etc). " +
			"The Validation section MUST end with a fenced bash block containing the exact command to verify the result."
	}

	systemPrompt := "You are a technical writer producing user-executable documentation. " +
		"Write steps as concrete user actions, not feature descriptions. " +
		"Return your result encoded as a single JSON object. Do not wrap the entire response in a markdown fence; markdown inside JSON string fields is allowed and expected."

	userPrompt := fmt.Sprintf(
		"Decision status: %s\n"+
			"Doc profile kind: %s | Required sections: %s\n"+
			"Audience: %s | Tone: %s\n\n"+
			"Content rules:\n"+
			"1. Steps tell the user exactly what to run or edit.\n"+
			"2. %s\n"+
			"3. Inside code blocks, add a short comment only when a value is non-obvious; do not fabricate values not supported by the evidence.\n"+
			"4. If evidence is insufficient for a full runnable example, list what is unknown in warnings instead of inventing details.\n\n"+
			"Return a single JSON object with this shape:\n"+
			"{\"delta\":{\"target_doc_ref\":\"string\",\"patch_type\":\"section_replace|add_section|remove_section|note_fix\",\"changed_sections\":[\"string\"],\"changes_markdown\":\"string\"},"+
			"\"document\":{\"doc_kind\":\"string\",\"title\":\"string\",\"summary\":\"string\",\"body_markdown\":\"string\",\"tags\":[\"string\"]},"+
			"\"warnings\":[\"string\"]}\n\n"+
			"JSON encoding rules:\n"+
			"- status update_required: populate delta; set document to {}.\n"+
			"- status new_document_required: populate document; set delta to {}.\n"+
			"- status no_changes_required: set both delta and document to {}.\n"+
			"- Newlines inside string values MUST be encoded as \\n. Triple-backtick fences are required in Steps/Validation content for update_required and new_document_required outputs.\n\n"+
			"## Original Query\n%s\n\n"+
			"## Extracted Facts JSON\n%s\n\n"+
			"## Audit JSON\n%s\n\n"+
			"## Existing Documentation Chunks\n%s\n\n"+
			"## Existing Generated Documentation Chunks\n%s",
		decision.Status,
		profile.Kind,
		strings.Join(profile.RequiredSections, ", "),
		profile.Audience,
		profile.Tone,
		codeBlockRule,
		req.QueryText,
		extractedFactsJSON,
		auditJSON,
		docCtx,
		genDocCtx,
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

func isInstructionalRequest(queryText string) bool {
	query := strings.ToLower(strings.TrimSpace(queryText))
	if query == "" {
		return false
	}
	for _, kw := range []string{
		"how to", "how do i", "setup", "configure", "configuration",
		"command", "script", "snippet", "example", "yaml", "json",
		"curl", "steps", "install", "run", "execute", "integration",
		"write", "create", "generate", "show", "share",
	} {
		if strings.Contains(query, kw) {
			return true
		}
	}
	return false
}

func buildDocRepairPrompt(stepName, schemaHint, invalidOutput string) []Message {
	systemPrompt := "You repair invalid JSON. Return only valid JSON and preserve meaning."
	userPrompt := fmt.Sprintf(
		"Step: %s\n"+
			"Schema hint:\n%s\n\n"+
			"Invalid output:\n%s\n\n"+
			"Return corrected JSON only.",
		stepName,
		schemaHint,
		invalidOutput,
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}
