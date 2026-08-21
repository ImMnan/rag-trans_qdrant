package pipeline

const (
	defaultModelContextTokens = 32768
	defaultSafetyTokens       = 2048

	minAutoBudgetTokens = 256
	maxAutoBudgetTokens = 8192

	minRequestTokenLimit = 64
	maxRequestTokenLimit = 12288

	// maxContextCharsPerSide caps retrieved chunk content (change or code side) before prompting,
	// as a hard backstop since the ~4-chars/token estimate below can undercount dense code text.
	maxContextCharsPerSide = 30000
)

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
	case "extract", "audit":
		multiplier = 0.6
	case "repair":
		multiplier = 0.45
	case "generate":
		multiplier = 1.0
	}

	stepBudget := int(float64(base) * multiplier)
	return clampInt(stepBudget, minAutoBudgetTokens, maxAutoBudgetTokens)
}

func estimateMessageTokens(messages []Message) int {
	if len(messages) == 0 {
		return 0
	}

	tokens := 0
	for _, m := range messages {
		// Rough approximation: ~4 chars/token + chat formatting overhead.
		tokens += (len(m.Content) / 4) + 6
	}

	// Small fixed overhead for request framing.
	return tokens + 12
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
