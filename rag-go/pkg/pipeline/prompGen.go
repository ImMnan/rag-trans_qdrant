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

const docAuthorityRules = "Evidence authority rules:\n" +
	"- Source code and change evidence is the ground truth for anything within the scope of the Original Query. Existing documentation is a candidate artifact to be verified against it, never the other way round.\n" +
	"- When documentation and code disagree on a query-scoped point, the code wins: use the code value and explicitly call the documentation outdated.\n" +
	"- When no documentation matches, or the matched documentation is about a different topic, ignore it entirely and answer from the code evidence. Never bend the answer to fit an unrelated document.\n" +
	"- Documentation may still supply context the code cannot show (intent, prerequisites, ownership, external systems). Use it for that, and label it as documented rather than verified.\n" +
	"- Do not use implementation details outside the Original Query to mark documentation stale or conflicting; stay within the query scope.\n" +
	"- Where the code evidence is genuinely absent or ambiguous, say so in warnings/unknowns instead of inventing a value or silently trusting the documentation.\n"

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

	queryText := strings.TrimSpace(req.QueryText)
	isInstructional := isInstructionalRequest(queryText)
	isSupportQuestion := isSupportQuestion(queryText)

	directAnswerGuidance := "You are answering as the end-user of this application, not as a developer maintaining the codebase. Use the repository profile as the product use case and answer only what a user needs to do to achieve the stated goal. Base your guidance on the provided application profile, config files, and runtime evidence. Only use environment variables, file paths, manifests, ports, commands, and settings that are visible in the provided context; do not invent new values. If the repo is Kubernetes or Docker based, provide commands, manifests, env vars, volume mounts, ports, and health checks that users actually run. Do not propose changes to source code, Dockerfiles, functions, templates, or implementation internals. Do not print function signatures, patch diffs, or code changes. If a required value is not visible in the provided context, say 'Unknown based on provided context' and do not invent it."
	if isSupportQuestion {
		directAnswerGuidance = "Answer the question directly from the provided repository evidence. For factual support questions, give a concise yes/no or supported/unsupported answer and explain the evidence in plain language. Do not force the answer into a how-to workflow or step-by-step usage guide unless the user explicitly asked for instructions. If the evidence does not show support, say so clearly and avoid guessing. If nothing in the provided context supports a claim, return 'No evidence in the provided context supports this claim.'"
	} else if !isInstructional {
		directAnswerGuidance = "Answer as the operator or user of the application. Focus on the exact usage required to achieve the goal, not on how the software is implemented internally. Use the application profile as the product context. Prefer runtime configuration, manifests, commands, and files a user actually executes. Only use values visible in the provided context; do not invent env vars or config keys. Do not propose source code changes, function edits, Dockerfile refactors, or implementation details. If nothing supported by the evidence matches the request, return 'No usage guidance found for this requirement in the provided context.'"
	}
	appProfileCtx := "No application profile is configured for this repository."
	if req.AppProfile != "" {
		appProfileCtx = req.AppProfile
	}

	directPrompt := fmt.Sprintf(
		"Role: answer the question using the repository evidence, not as an engineer changing the code. "+
			"Treat the application profile as product context, but do not force procedural steps when the request is a direct factual question. "+
			"If the question is about support, capability, or existence, answer directly from the provided evidence and state whether it is supported, unsupported, or unconfirmed. "+
			"Only provide numbered steps or fenced command blocks when the user is explicitly asking for instructions or usage guidance. "+
			"Only use values visible in the provided context. Never invent env vars, config keys, or file paths. "+
			"Never suggest code edits, source patches, Dockerfile rewrites, function printing, or implementation-level changes. "+
			"If the request is not supported by the provided evidence, return a clear evidence-based answer such as 'No evidence in the provided context supports this claim.' or 'No usage guidance found for this requirement in the provided context.' when it is truly a procedural question.\n\n"+
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
		"The code facts are the ground truth; your job is to judge the documentation against them, not to defend it. " +
		"Use only the provided facts and doc context. " +
		"Return strict JSON and do not add markdown fences."

	userPrompt := fmt.Sprintf(
		"Return JSON with this shape only:\n"+
			"{\"matched_docs\":[{\"doc_ref\":string,\"title\":string,\"why_matched\":string,\"match_confidence\":number}],\"missing_facts\":[string],\"conflicting_facts\":[string],\"stale_facts\":[string],\"summary\":string}\n\n"+
			"Rules:\n"+
			"- Only match a document that actually covers the subject of the Original Query. A shared keyword, a shared product name, or generic overlap is NOT a match.\n"+
			"- Returning an empty matched_docs array is the correct answer when nothing genuinely covers the query. Never pad the list with the closest available document.\n"+
			"- match_confidence range must be 0.0 to 1.0. Score below 0.75 unless the document clearly covers the query topic; anything below that is discarded downstream.\n"+
			"- Treat the extracted code facts as ground truth for the Original Query. Where a matched document states something different, list it under conflicting_facts or stale_facts rather than accepting the document.\n"+
			"- List under missing_facts any query-scoped code fact the matched documentation does not cover.\n"+
			"- Do not mark documentation missing, conflicting, or stale for facts outside the Original Query, even if those facts appear in code.\n"+
			"- Each documentation chunk is prefixed with \"[source: <filename>]\" on its own line. "+
			"When a chunk matches the query topic, copy that exact filename (e.g. \"docs/gatling.md\") into the doc_ref field. "+
			"Never invent a doc_ref and never leave it empty when a [source: ...] label is present in the matched chunk.\n\n"+
			"%s\n"+
			"## Original Query\n%s\n\n"+
			"## Extracted Facts JSON (ground truth)\n%s\n\n"+
			"## Existing Documentation Chunks\n%s\n\n"+
			"## Existing Generated Documentation Chunks\n%s",
		docAuthorityRules, req.QueryText, extractedFactsJSON, docCtx, genDocCtx,
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

// docGenerateSchemaForStatus returns only the branch of the result the decided status
// uses. Asking for the full three-branch object invites the model into fields it must
// then leave empty, which is where the encoding usually breaks.
func docGenerateSchemaForStatus(status DocDecisionStatus) string {
	const (
		deltaShape    = `"delta":{"target_doc_ref":"<doc ref>","patch_type":"section_replace|add_section|remove_section|note_fix","changed_sections":["<section name>"],"changes_markdown":"<ready-to-paste markdown>"}`
		documentShape = `"document":{"doc_kind":"kb_article","title":"<title>","summary":"<one paragraph>","body_markdown":"<full document markdown>","tags":["<tag>"]}`
		instrShape    = `"resolved_instructions":{"body_markdown":"<full step-by-step markdown>","corrections":["<correction note>"]}`
		warningsShape = `"warnings":["<warning>"]`
	)

	switch status {
	case StatusUpdateRequired:
		return "{" + deltaShape + "," + instrShape + "," + warningsShape + "}"
	case StatusNewDocumentRequired:
		return "{" + documentShape + "," + warningsShape + "}"
	default:
		return "{" + instrShape + "," + warningsShape + "}"
	}
}

// productUserAudienceRules keep generated documentation aimed at whoever uses the product,
// rather than at an engineer maintaining its source.
const productUserAudienceRules = "Audience rules:\n" +
	"- You are writing for the USER of this product, not for an engineer maintaining its source. The Application Profile is the product use case; treat it as who the reader is and what they are trying to accomplish.\n" +
	"- Source code is your evidence for what the product actually does. It is never the subject of the document. The reader cannot edit it.\n" +
	"- Write what the reader runs, configures, or deploys: commands, config files, manifests, env vars, flags, ports, endpoints, and how to confirm it worked.\n" +
	"- Never instruct the reader to modify source code, functions, Dockerfiles, or templates, and never present a patch, diff, or function signature as a step.\n" +
	"- Do not describe the change history or how a feature was implemented. Describe how to use the feature as it behaves today.\n" +
	"- Only use values visible in the provided context. If a value the reader needs is not visible, say 'Unknown based on provided context' and record it in warnings rather than inventing it.\n"

func buildDocGeneratePrompt(
	req Request,
	decision DocDecision,
	extractedFactsJSON string,
	auditJSON string,
	profile DocProfile,
	changeChunks []string,
	codeChunks []string,
	docChunks []string,
	genDocChunks []string,
) []Message {
	evidenceCtx := joinChunks(mergeEvidenceChunks(changeChunks, codeChunks), "No change or code context found.")
	docCtx := joinChunks(docChunks, "No documentation context found.")
	genDocCtx := joinChunks(genDocChunks, "No generated documentation context found.")
	appProfileCtx := "No application profile is configured for this repository."
	if strings.TrimSpace(req.AppProfile) != "" {
		appProfileCtx = req.AppProfile
	}

	codeBlockRule := "In Steps and Validation sections, each actionable step must include at least one triple-backtick fenced code block with the correct language tag (bash, yaml, json, go, etc). The Validation section must end with a fenced bash block containing the exact verification command(s)."
	if isInstructionalRequest(req.QueryText) {
		codeBlockRule = "For every numbered step that involves a file, option, flag, or command: " +
			"write the step description, then on the very next line output a triple-backtick fenced code block " +
			"using the correct language tag (yaml, bash, json, go, etc). " +
			"The Validation section MUST end with a fenced bash block containing the exact command to verify the result."
	}

	systemPrompt := "You are a technical writer producing end-user documentation for a product. " +
		"Your reader is a user or operator of the product described in the Application Profile, never an engineer changing its source. " +
		"Write steps as concrete actions that reader performs, not as feature descriptions or implementation notes. " +
		"The Source Code And Change Evidence section is the authoritative truth about how the product behaves: when it contradicts the existing documentation, follow the code and say so. " +
		"Never copy the angle-bracket placeholders from the output shape into your answer; every field must carry real content derived from the evidence. " +
		"Your entire response is one JSON object: it begins with { and ends with }, with no surrounding text and no markdown fence. " +
		"Markdown belongs inside the JSON string values, where line breaks are written as backslash-n."

	userPrompt := fmt.Sprintf(
		"Decision status: %s\n"+
			"Doc profile kind: %s | Required sections: %s\n"+
			"Audience: %s | Tone: %s\n\n"+
			"Content rules:\n"+
			"1. Steps tell the reader exactly what to run or edit in their own environment.\n"+
			"2. %s\n"+
			"3. Inside code blocks, add a short comment only when a value is non-obvious; do not fabricate values not supported by the evidence.\n"+
			"4. If evidence is insufficient for a full runnable example, list what is unknown in warnings instead of inventing details.\n"+
			"5. Preserve existing documentation for any use-case detail outside the Original Query or not 100%% clearly changed by code/change evidence.\n"+
			"6. When no document was matched, write the answer purely from the Source Code And Change Evidence; do not borrow structure or claims from unrelated documentation chunks.\n"+
			"7. Values that appear in the Source Code And Change Evidence outrank the same value written in the existing documentation; cite the file path when you rely on code evidence.\n"+
			"8. Every markdown field you populate must be real content. Returning the literal placeholder \"string\", or an empty body for the field required by the decision status, is an invalid answer.\n\n"+
			"%s\n"+
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
			"- body_markdown must follow the same numbered-step and fenced-code-block requirements as Steps/Validation elsewhere in this prompt.\n\n"+
			"OUTPUT FORMAT — read carefully, this is the whole response:\n"+
			"Emit exactly this JSON object and nothing else, with these keys and no others:\n"+
			"%s\n\n"+
			"JSON encoding rules:\n"+
			"- Output starts with { and ends with }. No prose before it, no prose after it, no markdown fence around it.\n"+
			"- Inside a string value, every line break MUST be written as the two characters backslash-n. Never press Enter inside a string.\n"+
			"- Inside a string value, every double quote MUST be written as backslash-quote, and every backslash as double-backslash.\n"+
			"- Triple-backtick fences are plain characters and need no escaping; write ```bash directly inside the string.\n"+
			"- No trailing comma before } or ].\n"+
			"- Do not add keys that are not listed above, and do not omit any that are.\n\n"+
			"## Original Query\n%s\n\n"+
			"## Application Profile (who the reader is and what this product is for)\n%s\n\n"+
			"## Source Code And Change Evidence (authoritative)\n%s\n\n"+
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
		productUserAudienceRules,
		docAuthorityRules,
		docGenerateSchemaForStatus(decision.Status),
		req.QueryText,
		appProfileCtx,
		evidenceCtx,
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

func isSupportQuestion(queryText string) bool {
	query := strings.ToLower(strings.TrimSpace(queryText))
	if query == "" {
		return false
	}
	for _, kw := range []string{
		"do we support", "does it support", "is support", "supported",
		"is there support", "can it", "possible to", "do you support",
		"does this support", "is x supported", "support for",
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
			"Rules:\n"+
			"- The schema hint describes field names and types only. Never copy its placeholder values "+
			"(\"string\", \"kb_article\", the pipe-separated enum lists) into the repaired JSON.\n"+
			"- Keep every piece of real content from the invalid output, including full markdown bodies.\n"+
			"- If the invalid output was cut off mid-value, close the structure without discarding the text already produced.\n\n"+
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
