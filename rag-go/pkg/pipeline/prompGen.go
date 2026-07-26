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
			"Use only the provided context. Structure your answer with these sections:\n" +
			"1. **What Changed** - describe the commits/diffs concisely.\n" +
			"2. **User Impact** - explain what end-users will notice or need to act on.\n" +
			"3. **Security & Performance** - flag any security fixes or performance optimizations; " +
			"write 'None identified' if absent."

		userPrompt := fmt.Sprintf("## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Request\n%s",
			changeCtx, codeCtx, req.QueryText)

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
			"- match_confidence range must be 0.0 to 1.0.\n\n"+
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
	exampleRequirement := buildDocExampleRequirement(req, profile)

	systemPrompt := "You generate documentation artifacts from validated evidence. " +
		"Return strict JSON only, with no markdown fence or extra text."

	userPrompt := fmt.Sprintf(
		"Decision status: %s\n"+
			"Doc profile kind: %s\n"+
			"Required sections: %s\n"+
			"Audience: %s\n"+
			"Tone: %s\n"+
			"Step style: %s\n"+
			"Example policy: %s\n"+
			"Validation policy: %s\n\n"+
			"Return JSON with this shape only:\n"+
			"{\"delta\":{\"target_doc_ref\":string,\"patch_type\":\"section_replace|add_section|remove_section|note_fix\",\"changed_sections\":[string],\"changes_markdown\":string},\"document\":{\"doc_kind\":string,\"title\":string,\"summary\":string,\"body_markdown\":string,\"tags\":[string]},\"warnings\":[string]}\n\n"+
			"Rules:\n"+
			"- If status is update_required: populate delta and keep document empty object.\n"+
			"- If status is new_document_required: populate document and keep delta empty object.\n"+
			"- If status is no_changes_required: keep both delta and document empty objects.\n"+
			"- The document format must follow kb_article style for now.\n"+
			"- The Steps section must be actionable. Each step should tell the user exactly what to do, not just describe the feature.\n"+
			"- When the request asks for setup, integration, configuration, commands, scripts, examples, snippets, YAML, JSON, or how-to guidance, include at least one fenced code block inside body_markdown or changes_markdown using the most relevant language tag.\n"+
			"- Use code comments inside examples when they help the user map values back to the documented context, but do not invent values that are unsupported by the evidence.\n"+
			"- Prefer short executable examples over abstract pseudocode.\n"+
			"- The Validation section must include a concrete command, request, or observable verification step when supported by the evidence.\n"+
			"- If the evidence does not support a full runnable example, say what is unknown in warnings instead of fabricating details.\n"+
			"- Keep body_markdown as plain markdown text in the JSON string; do not wrap the entire field in markdown fences.\n"+
			"- Specific example requirement: %s\n\n"+
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
		profile.StepStyle,
		profile.ExamplePolicy,
		profile.ValidationPolicy,
		exampleRequirement,
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

func buildDocExampleRequirement(req Request, profile DocProfile) string {
	query := strings.ToLower(strings.TrimSpace(req.QueryText))
	if query == "" {
		return profile.ExamplePolicy
	}

	keywords := []string{
		"how to",
		"how do i",
		"setup",
		"configure",
		"configuration",
		"command",
		"script",
		"snippet",
		"example",
		"yaml",
		"json",
		"curl",
		"steps",
		"install",
		"run",
		"execute",
		"integration",
	}

	for _, keyword := range keywords {
		if strings.Contains(query, keyword) {
			return "This request is instructional. Include at least one runnable fenced example and one concrete validation command if the retrieved evidence supports them."
		}
	}

	return "Include fenced examples only when they materially help explain supported evidence."
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
