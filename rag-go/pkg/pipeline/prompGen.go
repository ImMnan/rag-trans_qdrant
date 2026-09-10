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
	"- Source code and change evidence is the ground truth for anything in scope of the Original Query. Existing documentation is a candidate artifact to verify against it, never the reverse.\n" +
	"- Where documentation and code disagree in scope, the code wins: use the code value, cite its file path, and call the documentation outdated.\n" +
	"- Where no documentation matches, or it covers a different topic, ignore it and answer from the code. Never bend the answer to fit an unrelated document.\n" +
	"- Documentation may still supply what code cannot show (intent, prerequisites, external systems). Use it for that, labelled as documented rather than verified.\n" +
	"- Keep to the query scope: do not mark documentation stale over implementation details the query never asked about.\n" +
	"- Where the evidence is absent or ambiguous, put it in warnings instead of inventing a value or silently trusting the documentation.\n"

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

// productUserAudienceRules keep generated documentation aimed at whoever uses the product,
// rather than at an engineer maintaining its source.
const productUserAudienceRules = "Audience rules:\n" +
	"- Write for the USER of this product. The Application Profile says who they are and what they are trying to accomplish.\n" +
	"- Source code is evidence of what the product does, never the subject of the document. The reader cannot edit it.\n" +
	"- Write what the reader runs, configures, or deploys: commands, config files, manifests, env vars, flags, ports, endpoints, and how to confirm it worked.\n" +
	"- Never tell the reader to modify source, functions, Dockerfiles, or templates, and never present a patch, diff, or function signature as a step.\n" +
	"- Describe the feature as it behaves today, not its change history or how it was implemented.\n" +
	"- Use only values visible in the provided context. If one the reader needs is missing, write 'Unknown based on provided context' and note it in warnings.\n"

// buildDocTriagePrompt asks which retrieved documentation chunks answer the query. It sees
// documentation only: this step decides relevance and coverage, never correctness.
func buildDocTriagePrompt(req Request, indexedDocChunks []string) []Message {
	systemPrompt := "You triage retrieved documentation against a user's question. " +
		"You decide which chunks are worth keeping and how much of the question they answer. " +
		"You are not judging whether the documentation is correct, only whether it is on topic and complete. " +
		"Your entire response is one JSON object beginning with { and ending with }, with no surrounding text and no markdown fence."

	userPrompt := fmt.Sprintf(
		"Return exactly this JSON object:\n"+
			"{\"topic\":\"<topic>\",\"coverage\":\"complete|partial|none\","+
			"\"relevant_chunks\":[{\"index\":0,\"relevance\":0.0,\"why\":\"<why it is relevant>\"}],"+
			"\"missing_points\":[\"<part of the question the chunks do not answer>\"],\"reason\":\"<one sentence>\"}\n\n"+
			"Rules:\n"+
			"- Each chunk below opens with '[chunk N]'. Reference chunks only by that integer N. Never invent an index or cite a filename.\n"+
			"- Keep a chunk only if it helps answer this specific question. A shared keyword or product name is not relevance.\n"+
			"- An empty relevant_chunks array with coverage 'none' is the correct answer when nothing on this list is on topic. Never pad it with the closest available chunk.\n"+
			"- relevance is 0.0 to 1.0. Score below %.2f for anything you would not want quoted in the answer; those are discarded.\n"+
			"- coverage 'complete' means the kept chunks answer the whole question, 'partial' means they answer some of it, 'none' means they do not address it.\n"+
			"- missing_points lists what the question asks but these chunks do not cover. Leave it empty only when coverage is 'complete'.\n"+
			"- Inside a string, write line breaks as backslash-n. No trailing comma before } or ].\n\n"+
			"## Question\n%s\n\n"+
			"## Retrieved Documentation Chunks\n%s",
		minChunkRelevance, req.QueryText, joinChunks(indexedDocChunks, "No documentation chunks were retrieved."),
	)

	return []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

// buildDocComposePrompt writes the answer. Source evidence is always present so the model
// can correct whatever the surviving documentation gets wrong.
func buildDocComposePrompt(
	req Request,
	profile DocProfile,
	triage DocTriageResult,
	keptDocChunks []string,
	changeChunks []string,
	codeChunks []string,
) []Message {
	evidenceCtx := joinChunks(mergeEvidenceChunks(changeChunks, codeChunks), "No change or code context found.")
	appProfileCtx := "No application profile is configured for this repository."
	if strings.TrimSpace(req.AppProfile) != "" {
		appProfileCtx = req.AppProfile
	}

	codeBlockRule := "Every actionable step carries a triple-backtick fenced code block with the correct language tag (bash, yaml, json, etc), and the Validation section ends with a fenced bash block containing the exact verification command."
	if isInstructionalRequest(req.QueryText) {
		codeBlockRule = "For every numbered step involving a file, option, flag, or command: write the step, then on the very next line a triple-backtick fenced code block with the correct language tag. The Validation section MUST end with a fenced bash block containing the exact command to verify the result."
	}

	baselineRules := "Baseline rules (documentation was found for this query):\n" +
		"- Build the answer on the documentation baseline below, then check every step against the source evidence before you keep it.\n" +
		"- Where the evidence contradicts a documented step or value, use the evidence and append an inline note '(Corrected: doc said X; source shows Y)' to that step.\n" +
		"- Record each such fix as one sentence in corrections. Leave corrections empty only if nothing needed changing.\n" +
		"- Fill any gap listed under Gaps To Close from the source evidence; if the evidence cannot close it, say so in warnings.\n"
	docBaselineCtx := joinChunks(keptDocChunks, "")
	if len(keptDocChunks) == 0 {
		baselineRules = "Fresh document rules (no documentation covers this query):\n" +
			"- Write the answer entirely from the source evidence below.\n" +
			"- corrections stays empty: there is no existing documentation to correct.\n" +
			"- Do not speculate beyond the evidence; put anything you cannot establish in warnings.\n"
		docBaselineCtx = "No relevant documentation exists for this query."
	}

	gaps := "None recorded."
	if len(triage.MissingPoints) > 0 {
		gaps = "- " + strings.Join(triage.MissingPoints, "\n- ")
	}

	systemPrompt := "You are a technical writer producing end-user documentation for a product. " +
		"Your reader is a user or operator of the product described in the Application Profile, never an engineer changing its source. " +
		"Your entire response is one JSON object: it begins with { and ends with }, with no surrounding text and no markdown fence. " +
		"Markdown belongs inside the JSON string values, where line breaks are written as backslash-n."

	userPrompt := fmt.Sprintf(
		"Documentation coverage of this query: %s\n"+
			"Doc profile kind: %s | Required sections: %s\n"+
			"Audience: %s | Tone: %s\n\n"+
			"Content rules:\n"+
			"1. body_markdown is the complete answer the reader follows, covering the required sections in order.\n"+
			"2. %s\n"+
			"3. Add a comment inside a code block only when a value is non-obvious; never fabricate a value the evidence does not support.\n"+
			"4. title names the task the reader is accomplishing, not the code that implements it.\n"+
			"5. Every field you populate must carry real content. An angle-bracket placeholder or an empty body is an invalid answer.\n\n"+
			"%s\n"+
			"%s\n"+
			"%s\n"+
			"OUTPUT FORMAT — this is the whole response. Emit exactly this JSON object, with these keys and no others:\n"+
			"{\"title\":\"<title>\",\"body_markdown\":\"<full markdown>\",\"corrections\":[\"<correction note>\"],\"warnings\":[\"<warning>\"]}\n\n"+
			"JSON encoding rules:\n"+
			"- Start at { and end at }. No prose either side, no markdown fence.\n"+
			"- Inside a string, write every line break as backslash-n, every double quote as backslash-quote, every backslash as double-backslash. Never press Enter inside a string.\n"+
			"- Triple-backtick fences are plain characters and need no escaping; write ```bash directly inside the string.\n"+
			"- No trailing comma before } or ].\n\n"+
			"## Question\n%s\n\n"+
			"## Application Profile (who the reader is and what this product is for)\n%s\n\n"+
			"## Gaps To Close\n%s\n\n"+
			"## Source Code And Change Evidence (authoritative)\n%s\n\n"+
			"## Documentation Baseline (already filtered to the relevant chunks)\n%s",
		triage.Coverage,
		profile.Kind,
		strings.Join(profile.RequiredSections, ", "),
		profile.Audience,
		profile.Tone,
		codeBlockRule,
		productUserAudienceRules,
		docAuthorityRules,
		baselineRules,
		req.QueryText,
		appProfileCtx,
		gaps,
		evidenceCtx,
		docBaselineCtx,
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
