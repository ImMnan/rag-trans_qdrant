package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// loadApplicationProfile returns the configured mounted profile for repoID.
func loadApplicationProfile(profileDir, profileFile string) (string, error) {
	if strings.TrimSpace(profileDir) == "" || strings.TrimSpace(profileFile) == "" {
		return "", nil
	}

	profilePath := filepath.Join(profileDir, filepath.Base(profileFile))
	content, err := os.ReadFile(profilePath)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("read application profile %q: %w", profilePath, err)
	}

	return string(content), nil
}
