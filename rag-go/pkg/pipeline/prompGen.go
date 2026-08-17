package pipeline

import (
	"fmt"
	"strings"
)

// buildPrompt assembles the LLM messages from retrieved chunks.
func buildPrompt(req Request, changeChunks, codeChunks []string) []Message {
	changeCtx := joinChunks(changeChunks, "No change data found.")
	codeCtx := joinChunks(codeChunks, "No source context found.")

	if strings.EqualFold(strings.TrimSpace(req.Type), "standard") {
		systemPrompt := "You are a senior engineer producing product release summaries. " +
			"Use only the provided context. The Diff / Change Hunks context has already been filtered to the requested reporting window. " +
			"Treat the Request as topic and do not apply additional date filtering.\n\n" +
			"Structure your answer with these sections:\n" +
			"0. **What Changed** - describe the provided commits/diffs concisely. " +
			"If no changes are provided, write exactly: 'No changes found in the requested timeframe based on provided context.'\n" +
			"1. **User Impact** - explain what end-users will notice or need to act on.\n" +
			"2. **Security & Performance** - flag any security fixes or performance optimizations; " +
			"write 'None identified' if absent.\n"

		userPrompt := fmt.Sprintf("## Reporting Window\n%s to %s\n\n## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Request\n%s",
			req.FromDate, req.ToDate, changeCtx, codeCtx, req.QueryText)

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
	evidenceCtx := joinChunks(mergeEvidenceChunks(changeChunks, codeChunks), "No change or code context found.")

	systemPrompt := "You extract product and implementation facts from code evidence. " +
		"Use only the provided change/code context. " +
		"Return strict JSON and do not add markdown fences."

	userPrompt := fmt.Sprintf(
		"Return JSON with this shape only:\n"+
			"{\"topic\":string,\"facts\":[{\"fact\":string,\"source\":\"change|code\",\"confidence\":number}],\"unknowns\":[string]}\n\n"+
			"Rules:\n"+
			"- Extract factual statements only from evidence.\n"+
			"- If uncertain, list the point under unknowns.\n"+
			"- confidence range must be 0.0 to 1.0.\n"+
			"- Each evidence chunk below is prefixed with \"[source: change]\" or \"[source: code]\" on its own line; "+
			"copy that exact label into the fact's source field.\n\n"+
			"## Original Query\n%s\n\n"+
			"## Original Answer Type\n%s\n\n"+
			"## Evidence (Diff/Change Hunks and Source Code, combined)\n%s",
		req.QueryText, req.Type, evidenceCtx,
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

// mergeEvidenceChunks squashes change and code chunks into one evidence list,
// tagging each chunk with its origin so the model can still cite source per fact.
func mergeEvidenceChunks(changeChunks, codeChunks []string) []string {
	merged := make([]string, 0, len(changeChunks)+len(codeChunks))
	for _, c := range changeChunks {
		merged = append(merged, "[source: change]\n"+c)
	}
	for _, c := range codeChunks {
		merged = append(merged, "[source: code]\n"+c)
	}
	return merged
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
			"Update patch rules (apply when status is update_required):\n"+
			"- changes_markdown is the actual ready-to-paste Markdown content for the change, not a description of the change.\n"+
			"- Never write a summary such as 'Added steps...' or 'Updated the documentation...' in changes_markdown.\n"+"- Use the existing documentation as the baseline and write complete replacement or insertion content for every changed section.\n"+"- Address every item in Audit JSON missing_facts, conflicting_facts, and stale_facts that is supported by the evidence.\n"+"- Also include each requested fact that the audit identifies as missing when it can be established from the extracted facts or source context.\n"+"- For patch_type add_section, changes_markdown must contain the complete new section, including its heading and detailed prose, steps, and code blocks where applicable.\n"+"- For patch_type section_replace, changes_markdown must contain the complete replacement section, including its heading; do not return only a list of changes.\n"+"- Put unsupported or unresolved items in warnings, but still write all supported details into changes_markdown.\n\n"+
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
