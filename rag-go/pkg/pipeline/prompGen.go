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

func buildPromptforDoc(req Request, changeChunks, codeChunks []string) []Message {
	changeCtx := joinChunks(changeChunks, "No change data found.")
	codeCtx := joinChunks(codeChunks, "No source context found.")

	if strings.EqualFold(strings.TrimSpace(req.Type), "standard") {
		systemPrompt := "You are a senior technical writer producing & updating product documentation. " +
			"Use only the provided context, consider these sections: \n" +
			"1. Is the change already documented? if yes, does it need updating? if yes, update it accordingly.\n" +
			"2. If the change is undocumented, create a new document\n" +
			"3. Use the existing format and style of the documentation.\n" +
			
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
