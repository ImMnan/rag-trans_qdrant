package pipeline

import (
	"strings"
)

// Evidence kinds describing what a retrieved code chunk actually contains.
const (
	EvidenceImplementation = "implementation"
	EvidenceComment        = "comment-only"
	EvidenceMixed          = "mixed"
)

// classifyCodeChunk reports whether a chunk carries executable code, only
// comments/docstrings, or both. Comment-only chunks are unverified author
// claims, not proof of behaviour.
func classifyCodeChunk(chunk string) string {
	var codeLines, commentLines int
	inBlock := false
	var blockCloser string

	for _, raw := range strings.Split(chunk, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "[source:") || strings.HasPrefix(line, "[evidence:") {
			continue
		}
		// Diff hunks carry +/- markers; classify the underlying line.
		if len(line) > 1 && (line[0] == '+' || line[0] == '-') && !strings.HasPrefix(line, "--") {
			line = strings.TrimSpace(line[1:])
			if line == "" {
				continue
			}
		}

		if inBlock {
			commentLines++
			if idx := strings.Index(line, blockCloser); idx >= 0 {
				inBlock = false
				if rest := strings.TrimSpace(line[idx+len(blockCloser):]); rest != "" {
					codeLines++
				}
			}
			continue
		}

		if closer, ok := blockCommentOpener(line); ok {
			commentLines++
			if !strings.Contains(line[len(closer):], closer) {
				inBlock = true
				blockCloser = closer
			}
			continue
		}

		if isLineComment(line) {
			commentLines++
			continue
		}
		codeLines++
	}

	switch {
	case codeLines == 0 && commentLines == 0:
		return EvidenceComment
	case codeLines == 0:
		return EvidenceComment
	case commentLines == 0:
		return EvidenceImplementation
	default:
		return EvidenceMixed
	}
}

func blockCommentOpener(line string) (string, bool) {
	for opener, closer := range map[string]string{
		`"""`:  `"""`,
		"'''":  "'''",
		"/*":   "*/",
		"<!--": "-->",
	} {
		if strings.HasPrefix(line, opener) {
			return closer, true
		}
	}
	return "", false
}

func isLineComment(line string) bool {
	for _, prefix := range []string{"//", "#", "*", "--", ";", "%"} {
		if strings.HasPrefix(line, prefix) {
			return true
		}
	}
	return false
}

// annotateCodeChunks tags each chunk with its evidence kind and returns the
// per-kind counts for logging and the response payload.
func annotateCodeChunks(chunks []string) ([]string, map[string]int) {
	annotated := make([]string, 0, len(chunks))
	counts := map[string]int{}
	for _, c := range chunks {
		kind := classifyCodeChunk(c)
		counts[kind]++
		annotated = append(annotated, "[evidence: "+kind+"]\n"+c)
	}
	return annotated, counts
}
