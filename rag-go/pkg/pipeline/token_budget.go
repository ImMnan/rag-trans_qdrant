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

	// Chunk budget weights split the shared pool 70/30 between source evidence and
	// documentation: code and change chunks are the authority for the answer, while docs
	// only need enough room to be matched and corrected.
	evidenceChunkWeight = 0.35 // change + code = 0.70
	docChunkWeight      = 0.15 // doc + generated doc = 0.30
)

// evidenceOnlyWeights is the weighting for paths that retrieve change and code only.
var evidenceOnlyWeights = []float64{evidenceChunkWeight, evidenceChunkWeight}

// docWorkflowWeights orders as change, code, doc, generated doc.
var docWorkflowWeights = []float64{evidenceChunkWeight, evidenceChunkWeight, docChunkWeight, docChunkWeight}

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
