package pipeline

import (
	"fmt"
	"strings"
)

// standardSectionHeadings are the exact heading lines a "standard" answer must contain, in order.
var standardSectionHeadings = []string{
	"**What Changed**",
	"**User Impact**",
	"**Security & Performance**",
}

const standardFormatContract = "Your answer MUST open with these three sections, reproduced verbatim including the ** markers, " +
	"with your content replacing the angle-bracket placeholders:\n\n" +
	"**What Changed**\n<concise description of the provided commits/diffs>\n\n" +
	"**User Impact**\n<what end-users will notice or need to act on>\n\n" +
	"**Security & Performance**\n<security fixes or performance optimizations>\n\n" +
	"Rules:\n" +
	"- Emit all three headings first, in this order, spelled exactly as shown. Never rename, translate, or reformat them.\n" +
	"- Do not add a preamble or a title before **What Changed**.\n" +
	"- If no changes are provided, the What Changed body must be exactly: " +
	"'No changes found in the requested timeframe based on provided context.'\n" +
	"- If no security or performance items apply, the Security & Performance body must be exactly: 'None identified'.\n" +
	"- The three sections above are the entire answer by default. Add a further section ONLY when the Request explicitly asks " +
	"for something those three cannot carry; name that section after what the Request asked for and write it in the style and tone it specifies. " +
	"If the Request asks nothing beyond a summary of changes, stop after Security & Performance and add nothing else.\n" +
	"- Never invent a section from the examples or wording of these instructions.\n"

const standardFormatReminder = "## Output Format\n" +
	"Begin with the three headings **What Changed**, **User Impact**, and **Security & Performance**, " +
	"in that order, spelled exactly like that — even if the Request implies a different structure. " +
	"Add a further section only if the Request explicitly asked for one; otherwise end after Security & Performance."

// evidenceGroundingRules tell the model how to read the [evidence: ...] labels
// that annotateCodeChunks attaches to each source chunk.
const evidenceGroundingRules = "Evidence grounding:\n" +
	"- Each Source / Doc Reference chunk starts with \"[evidence: implementation|mixed|comment-only]\" and \"[source: <file path>]\".\n" +
	"- Chunks are whole functions, so a mixed chunk is a doc comment plus its implementation. " +
	"Ground claims in the implementation body, not the doc comment. If the two disagree, the body wins and you must say so.\n" +
	"- comment-only chunks are author comments, docstrings, config, or prose, not proof that the code behaves that way. " +
	"Any claim resting only on such a chunk must be worded as 'documented as' or 'per comments in <file>', never as confirmed behaviour.\n" +
	"- A function being defined does not prove it is reachable or enabled. " +
	"Only call a capability supported if the context shows it invoked, registered, or configured; otherwise say it is defined but its use is not visible in the provided context.\n" +
	"- Cite the file path from the [source: ...] label when stating a technical fact.\n"

const docAuthorityRules = "Documentation authority rules:\n" +
	"- Code/change evidence is the paramount truth only for facts that directly answer the Original Query.\n" +
	"- Do not use unrelated implementation details to mark existing documentation stale, incomplete, or conflicting.\n" +
	"- When code evidence is incomplete, ambiguous, only comment-derived, or outside the query scope, preserve the existing documentation and put the uncertainty in warnings/unknowns.\n" +
	"- Existing documentation remains the fallback source for documented use-case context unless specific code/change evidence clearly disproves that documented point.\n" +
	"- A generated document or patch must stay within the documented use case and the Original Query; do not broaden the task because adjacent code exists.\n"

// buildPrompt assembles the LLM messages from retrieved chunks.
func buildPrompt(req Request, changeChunks, codeChunks []string) []Message {
	changeCtx := joinChunks(changeChunks, "No change data found.")
	codeCtx := joinChunks(codeChunks, "No source context found.")

	if strings.EqualFold(strings.TrimSpace(req.Type), "standard") {
		systemPrompt := "You are a senior engineer producing product release summaries. " +
			"Use only the provided context. The Diff / Change Hunks context has already been filtered to the requested reporting window. " +
			"Treat the Request as topic and do not apply additional date filtering. " +
			"Always answer the Request, but do so within the required section layout below.\n\n" +
			standardFormatContract + "\n" + evidenceGroundingRules

		userPrompt := fmt.Sprintf("## Reporting Window\n%s to %s\n\n## Diff / Change Hunks\n%s\n\n## Source / Doc Reference\n%s\n\n## Request\n%s\n\n%s",
			req.FromDate, req.ToDate, changeCtx, codeCtx, req.QueryText, standardFormatReminder)

		return []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		}
	}

	directAnswerGuidance := "You are answering as the end-user of this application, not as a developer maintaining the codebase. Use the repository profile as the product use case and answer only what a user needs to do to achieve the stated goal. Base your guidance on the provided application profile, config files, and runtime evidence. Only use environment variables, file paths, manifests, ports, commands, and settings that are visible in the provided context; do not invent new values. If the repo is Kubernetes or Docker based, provide commands, manifests, env vars, volume mounts, ports, and health checks that users actually run. Do not propose changes to source code, Dockerfiles, functions, templates, or implementation internals. Do not print function signatures, patch diffs, or code changes. If a required value is not visible in the provided context, say 'Unknown based on provided context' and do not invent it."
	if !isInstructionalRequest(req.QueryText) {
		directAnswerGuidance = "Answer as the operator or user of the application. Focus on the exact usage required to achieve the goal, not on how the software is implemented internally. Use the application profile as the product context. Prefer runtime configuration, manifests, commands, and files a user actually executes. Only use values visible in the provided context; do not invent env vars or config keys. Do not propose source code changes, function edits, Dockerfile refactors, or implementation details. If nothing supported by the evidence matches the request, return 'No usage guidance found for this requirement in the provided context.'"
	}
	appProfileCtx := "No application profile is configured for this repository."
	if req.AppProfile != "" {
		appProfileCtx = req.AppProfile
	}

	directPrompt := fmt.Sprintf(
		"Role: answer as a user of this application, not as an engineer changing the code. "+
			"The application profile is the product use case for this repository and should guide the answer. "+
			"Provide only practical usage steps for the user to achieve the goal. "+
			"Prefer numbered steps and fenced code blocks showing the exact YAML, commands, env vars, files, and verification commands the user is supposed to run. "+
			"Only use values visible in the provided context. Never invent env vars, config keys, or file paths. "+
			"Never suggest code edits, source patches, Dockerfile rewrites, function printing, or implementation-level changes. "+
			"If the request is not supported by the provided evidence, return 'No usage guidance found for this requirement in the provided context.'\n\n"+
			"%s\n\n"+
			"%s\n"+
			"## Application Profile\n%s\n\n## Diff / Change Hunks\n%s\n\n## Code Snapshot / Source Reference\n%s\n\n## Question\n%s",
		directAnswerGuidance, evidenceGroundingRules, appProfileCtx, changeCtx, codeCtx, req.QueryText,
	)

	return []Message{{Role: "user", Content: directPrompt}}
}

func joinChunks(chunks []string, fallback string) string {
	if len(chunks) == 0 {
		return fallback
	}
	return strings.Join(chunks, "\n---\n")
}

// hasStandardSections reports whether the answer carries all required headings in order.
func hasStandardSections(answer string) bool {
	cursor := 0
	for _, heading := range standardSectionHeadings {
		idx := strings.Index(answer[cursor:], heading)
		if idx < 0 {
			return false
		}
		cursor += idx + len(heading)
	}
	return true
}

// buildStandardFormatRepairPrompt reformats an off-template answer without re-running retrieval.
func buildStandardFormatRepairPrompt(answer string) []Message {
	systemPrompt := "You reformat an existing answer into a fixed template. " +
		"Preserve all factual content and wording as closely as possible. " +
		"Never add facts, and never drop facts. Only restructure.\n\n" +
		standardFormatContract

	userPrompt := "Reformat the following answer into the required layout. " +
		"Map existing content into the matching section; if one of the three required sections has no content, apply the fallback text from the rules. " +
		"Keep any material that does not belong to the three required sections as additional bolded sections after them, " +
		"but do not create sections that have no content in the answer being reformatted.\n\n" +
		"## Answer To Reformat\n" + answer

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
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
			"- Extract factual statements only from evidence and only when they directly answer the Original Query.\n"+
			"- Ignore adjacent implementation details that are not required to answer the Original Query.\n"+
			"- If uncertain, list the point under unknowns.\n"+
			"- If a point is only implied by comments or surrounding code and not 100%% clear from implementation/change evidence, list it under unknowns instead of facts.\n"+
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
			"- Treat code-derived facts as ground truth only for the specific Original Query.\n"+
			"- Do not mark documentation missing, conflicting, or stale for facts outside the Original Query, even if those facts appear in code.\n"+
			"- If extracted facts or source evidence do not 100%% clearly disprove the documented use case, keep the documentation valid and mention uncertainty in summary.\n"+
			"- Prefer a higher match_confidence for an existing document that covers the same use case, even when the code evidence has gaps.\n"+
			"- match_confidence range must be 0.0 to 1.0.\n"+
			"- Each documentation chunk is prefixed with \"[source: <filename>]\" on its own line. "+
			"When a chunk matches the query topic, copy that exact filename (e.g. \"docs/gatling.md\") into the doc_ref field. "+
			"Never leave doc_ref empty when a [source: ...] label is present in the matched chunk.\n\n"+
			"%s\n"+
			"## Original Query\n%s\n\n"+
			"## Extracted Facts JSON\n%s\n\n"+
			"## Existing Documentation Chunks\n%s\n\n"+
			"## Existing Generated Documentation Chunks\n%s",
		docAuthorityRules, req.QueryText, extractedFactsJSON, docCtx, genDocCtx,
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
			"4. If evidence is insufficient for a full runnable example, list what is unknown in warnings instead of inventing details.\n"+
			"5. Preserve existing documentation for any use-case detail outside the Original Query or not 100%% clearly changed by code/change evidence.\n"+
			"6. Do not replace valid documented use-case context with a generated interpretation unless Audit JSON shows a specific query-scoped conflict, stale fact, or missing fact.\n\n"+
			"%s\n"+
			"Update patch rules (apply when status is update_required):\n"+
			"- changes_markdown is the actual ready-to-paste Markdown content for the change, not a description of the change.\n"+
			"- Never write a summary such as 'Added steps...' or 'Updated the documentation...' in changes_markdown.\n"+"- Use the existing documentation as the baseline and write complete replacement or insertion content for every changed section.\n"+"- Address every item in Audit JSON missing_facts, conflicting_facts, and stale_facts that is supported by the evidence.\n"+"- Also include each requested fact that the audit identifies as missing when it can be established from the extracted facts or source context.\n"+"- For patch_type add_section, changes_markdown must contain the complete new section, including its heading and detailed prose, steps, and code blocks where applicable.\n"+"- For patch_type section_replace, changes_markdown must contain the complete replacement section, including its heading; do not return only a list of changes.\n"+"- Put unsupported or unresolved items in warnings, but still write all supported details into changes_markdown.\n\n"+
			"Resolved instructions rules (apply when status is update_required or no_changes_required AND a matched doc exists):\n"+
			"- resolved_instructions is what gets shown to the end user; it must be the FULL set of step-by-step instructions to accomplish the Original Query, not just the diff/patch.\n"+
			"- Start from the matched existing documentation as the baseline, then verify every step against the Extracted Facts / source evidence before including it.\n"+
			"- If a documented step, value, or claim conflicts with or is outdated relative to the evidence (see Audit JSON conflicting_facts/stale_facts), do NOT silently reproduce the old text: replace it with the corrected step and add a short inline note such as '(Corrected: doc said X; source shows Y)' right after that step.\n"+
			"- Also record every such fix as a short sentence in the corrections array, e.g. 'Step 3 previously said X; corrected to Y based on <file>.'\n"+
			"- If nothing needed correcting, still populate body_markdown with the full verified steps and leave corrections as an empty array.\n"+
			"- body_markdown must follow the same numbered-step and fenced-code-block requirements as Steps/Validation elsewhere in this prompt.\n"+
			"- Leave resolved_instructions.body_markdown empty only when status is new_document_required (the document field already carries the full content) or when no doc was matched at all.\n\n"+
			"Return a single JSON object with this shape:\n"+
			"{\"delta\":{\"target_doc_ref\":\"string\",\"patch_type\":\"section_replace|add_section|remove_section|note_fix\",\"changed_sections\":[\"string\"],\"changes_markdown\":\"string\"},"+
			"\"document\":{\"doc_kind\":\"string\",\"title\":\"string\",\"summary\":\"string\",\"body_markdown\":\"string\",\"tags\":[\"string\"]},"+
			"\"resolved_instructions\":{\"body_markdown\":\"string\",\"corrections\":[\"string\"]},"+
			"\"warnings\":[\"string\"]}\n\n"+
			"JSON encoding rules:\n"+
			"- status update_required: populate delta and resolved_instructions; set document to {}.\n"+
			"- status new_document_required: populate document; set delta and resolved_instructions to {}.\n"+
			"- status no_changes_required: set delta and document to {}; populate resolved_instructions when a doc was matched, otherwise leave it {}.\n"+
			"- Newlines inside string values MUST be encoded as \\n. Triple-backtick fences are required in Steps/Validation content for update_required and new_document_required outputs, and in resolved_instructions.body_markdown whenever it is populated.\n\n"+
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
		docAuthorityRules,
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
