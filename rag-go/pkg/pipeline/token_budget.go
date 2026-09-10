package pipeline

const (
	defaultModelContextTokens = 32768
	defaultSafetyTokens       = 2048

	minAutoBudgetTokens = 256
	maxAutoBudgetTokens = 8192

	minRequestTokenLimit = 64
	maxRequestTokenLimit = 12288

	// minGenerateOutputTokens is the room the generate step needs to emit a full document
	// or instruction body. Inputs are trimmed to protect it, otherwise a large retrieved
	// context starves the output budget down to minAutoBudgetTokens and the model returns
	// truncated JSON whose markdown fields come back empty after repair.
	minGenerateOutputTokens = 3072

	// charsPerToken is the rough ratio used throughout this file.
	charsPerToken = 4

	// promptOverheadTokens reserves room for the fixed prompt rules plus the facts and audit
	// JSON that sit alongside retrieved chunks in the largest doc-workflow call.
	promptOverheadTokens = 2000

	// maxContextCharsTotal is the combined budget for every retrieved chunk in one request.
	// It is a single shared pool rather than a per-side cap: the doc workflow has four sides,
	// and four independent caps would together exceed the whole model window.
	maxContextCharsTotal = (defaultModelContextTokens - defaultSafetyTokens - minGenerateOutputTokens - promptOverheadTokens) * charsPerToken
)

// AllocateChunkCharBudget distributes one shared character budget across the given chunk
// sides. Sides that need less than an equal share release the remainder to the others, so
// an empty side (for example generated docs) does not waste its allowance.
func AllocateChunkCharBudget(sides [][]string, totalChars int) [][]string {
	out := make([][]string, len(sides))
	pending := make([]int, 0, len(sides))
	for i, side := range sides {
		if len(side) == 0 {
			out[i] = side
			continue
		}
		pending = append(pending, i)
	}

	remaining := totalChars
	for len(pending) > 0 {
		share := remaining / len(pending)
		if share <= 0 {
			for _, i := range pending {
				out[i] = []string{}
			}
			break
		}

		stillPending := pending[:0:0]
		progressed := false
		for _, i := range pending {
			if chunksCharSize(sides[i]) > share {
				stillPending = append(stillPending, i)
				continue
			}
			// Fits entirely, so it only consumes what it needs and frees the rest.
			out[i] = sides[i]
			remaining -= chunksCharSize(sides[i])
			progressed = true
		}

		if !progressed {
			// Everyone left wants more than an equal share; split what is left evenly.
			for _, i := range stillPending {
				out[i] = TruncateChunksToCharBudget(sides[i], share)
			}
			break
		}
		pending = stillPending
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
	case "extract", "audit":
		multiplier = 0.6
	case "generate", "repair":
		// repair has to reproduce the full markdown it is fixing, so it gets the same room.
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
