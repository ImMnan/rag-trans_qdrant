package pipeline

import (
	"strings"
	"unicode/utf8"
)

const (
	defaultModelContextTokens = 32768
	defaultSafetyTokens       = 2048

	minAutoBudgetTokens = 256
	maxAutoBudgetTokens = 8192

	minRequestTokenLimit = 64
	maxRequestTokenLimit = 12288

	// minGenerateOutputTokens is the room the compose step needs to emit a full document
	// or instruction body. Inputs are trimmed to protect it, otherwise a large retrieved
	// context starves the output budget down to minAutoBudgetTokens and the model returns
	// truncated JSON whose markdown fields come back empty after repair.
	minGenerateOutputTokens = 3072

	// minTriageOutputTokens is the room triage needs. It emits indices and one-line reasons
	// rather than prose, so it needs far less than compose.
	minTriageOutputTokens = 1024

	// charsPerToken is the rough ratio used throughout this file.
	charsPerToken = 4

	// standardCharsPerToken is deliberately conservative: diffs tokenize denser than prose
	// (~3.7 chars/token observed), and charsPerToken under-counted enough to overflow vLLM.
	standardCharsPerToken = 3

	// promptOverheadTokens reserves room for the fixed prompt rules plus the facts and audit
	// JSON that sit alongside retrieved chunks in the largest doc-workflow call.
	promptOverheadTokens = 2000

	// maxContextCharsTotal is the combined budget for every retrieved chunk in one request.
	// It is a single shared pool rather than a per-side cap: the doc workflow has four sides,
	// and four independent caps would together exceed the whole model window.
	maxContextCharsTotal = (defaultModelContextTokens - defaultSafetyTokens - minGenerateOutputTokens - promptOverheadTokens) * charsPerToken

	// evidenceChunkWeight splits the shared pool evenly between change and code on paths
	// that put both in a single prompt. The doc workflow instead fits each of its two calls
	// separately, shedding documentation before source evidence in fitComposePrompt.
	evidenceChunkWeight = 0.5
)

// evidenceOnlyWeights is the weighting for paths that retrieve change and code only.
var evidenceOnlyWeights = []float64{evidenceChunkWeight, evidenceChunkWeight}

// AllocateChunkCharBudget distributes one shared character budget across the given chunk
// sides in proportion to weights. A side needing less than its share releases the remainder
// to the others, so an empty or small side (for example generated docs) is not wasted.
func AllocateChunkCharBudget(sides [][]string, weights []float64, totalChars int) [][]string {
	out := make([][]string, len(sides))
	active := make([]int, 0, len(sides))
	for i, side := range sides {
		out[i] = []string{}
		if len(side) == 0 || i >= len(weights) || weights[i] <= 0 {
			continue
		}
		active = append(active, i)
	}

	remaining := totalChars
	for len(active) > 0 && remaining > 0 {
		totalWeight := 0.0
		for _, i := range active {
			totalWeight += weights[i]
		}
		if totalWeight <= 0 {
			break
		}

		stillActive := make([]int, 0, len(active))
		progressed := false
		for _, i := range active {
			share := int(float64(remaining) * weights[i] / totalWeight)
			size := chunksCharSize(sides[i])
			if size <= share {
				// Fits entirely, so it consumes only what it needs and frees the rest.
				out[i] = sides[i]
				remaining -= size
				progressed = true
				continue
			}
			stillActive = append(stillActive, i)
		}

		if !progressed {
			// Everyone left wants more than its share; give each exactly its share.
			for _, i := range stillActive {
				share := int(float64(remaining) * weights[i] / totalWeight)
				out[i] = TruncateChunksToCharBudget(sides[i], share)
			}
			break
		}
		active = stillActive
	}

	return out
}

func chunksCharSize(chunks []string) int {
	total := 0
	for _, c := range chunks {
		total += len(c) + 4 // + separator overhead
	}
	return total
}

// TruncateChunksToCharBudget keeps chunks, in order, until adding the next one would
// exceed maxChars; it drops the remainder rather than cutting a chunk mid-content.
func TruncateChunksToCharBudget(chunks []string, maxChars int) []string {
	total := 0
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		total += len(c) + 4 // + separator overhead
		if total > maxChars {
			break
		}
		out = append(out, c)
	}
	return out
}

// ResolveTokenBudget returns the output-token budget for a single LLM call.
// If request token_limit is set, it is treated as a hard override (clamped).
func ResolveTokenBudget(req Request, messages []Message) int {
	if req.TokenLimit > 0 {
		return clampInt(req.TokenLimit, minRequestTokenLimit, maxRequestTokenLimit)
	}

	inputTokens := estimateMessageTokens(messages)
	available := defaultModelContextTokens - inputTokens - defaultSafetyTokens

	return clampInt(available, minAutoBudgetTokens, maxAutoBudgetTokens)
}

// ResolveDocStepTokenBudget returns a per-step budget for doc workflow calls.
// For explicit request overrides we keep the same budget across steps.
func ResolveDocStepTokenBudget(req Request, stepName string, messages []Message) int {
	base := ResolveTokenBudget(req, messages)
	if req.TokenLimit > 0 {
		return base
	}

	multiplier := 1.0
	switch stepName {
	case "triage":
		// Triage returns indices and short reasons, never prose.
		multiplier = 0.4
	case "compose", "repair":
		// repair has to reproduce the full markdown it is fixing, so it gets the same room.
		multiplier = 1.0
	}

	stepBudget := int(float64(base) * multiplier)
	return clampInt(stepBudget, minAutoBudgetTokens, maxAutoBudgetTokens)
}

func estimateMessageTokens(messages []Message) int {
	return estimateMessageTokensAt(messages, charsPerToken)
}

func estimateMessageTokensAt(messages []Message, charsPerTok int) int {
	if len(messages) == 0 {
		return 0
	}

	tokens := 0
	for _, m := range messages {
		// Content tokens + chat formatting overhead.
		tokens += (len(m.Content) / charsPerTok) + 6
	}

	// Small fixed overhead for request framing.
	return tokens + 12
}

// resolveStandardTokenBudget is ResolveTokenBudget using the conservative diff estimate.
func resolveStandardTokenBudget(req Request, messages []Message) int {
	if req.TokenLimit > 0 {
		return clampInt(req.TokenLimit, minRequestTokenLimit, maxRequestTokenLimit)
	}
	available := defaultModelContextTokens - estimateMessageTokensAt(messages, standardCharsPerToken) - defaultSafetyTokens
	return clampInt(available, minAutoBudgetTokens, maxAutoBudgetTokens)
}

// standardChunkTokenBudget is the room left for chunks once the fixed prompt and output are reserved.
func standardChunkTokenBudget(req Request, fixed []Message) int {
	output := minGenerateOutputTokens
	if req.TokenLimit > 0 {
		output = clampInt(req.TokenLimit, minRequestTokenLimit, maxRequestTokenLimit)
	}
	budget := defaultModelContextTokens - defaultSafetyTokens - output - estimateMessageTokensAt(fixed, standardCharsPerToken)
	return max(budget, minAutoBudgetTokens)
}

// packChunksByTokens groups chunks, in order, into batches that each fit budgetTokens.
// Nothing is dropped: a chunk larger than the budget is split at line boundaries.
// Always returns at least one (possibly empty) batch.
func packChunksByTokens(chunks []string, budgetTokens int) [][]string {
	budgetChars := budgetTokens * standardCharsPerToken
	batches := [][]string{}
	var current []string
	used := 0
	for _, chunk := range chunks {
		for _, piece := range splitChunkToChars(chunk, budgetChars) {
			size := len(piece) + 5 // + "\n---\n" separator
			if used+size > budgetChars && len(current) > 0 {
				batches = append(batches, current)
				current, used = nil, 0
			}
			current = append(current, piece)
			used += size
		}
	}
	if len(current) > 0 || len(batches) == 0 {
		batches = append(batches, current)
	}
	return batches
}

// splitChunkToChars splits an oversize chunk into consecutive pieces of at most maxChars,
// preferring line boundaries so diff hunks stay readable.
func splitChunkToChars(chunk string, maxChars int) []string {
	if len(chunk)+5 <= maxChars || maxChars <= 0 {
		return []string{chunk}
	}
	limit := maxChars - 5
	var pieces []string
	for len(chunk) > limit {
		cut := strings.LastIndexByte(chunk[:limit], '\n')
		if cut <= 0 {
			cut = limit
			for cut > 0 && !utf8.RuneStart(chunk[cut]) {
				cut--
			}
		}
		pieces = append(pieces, chunk[:cut])
		chunk = strings.TrimPrefix(chunk[cut:], "\n")
	}
	if chunk != "" {
		pieces = append(pieces, chunk)
	}
	return pieces
}

func clampInt(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}
