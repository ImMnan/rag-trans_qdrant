package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func testSubmission(t *testing.T) FeedbackSubmission {
	t.Helper()
	var submission FeedbackSubmission
	if err := json.Unmarshal([]byte(standardPayload), &submission); err != nil {
		t.Fatal(err)
	}
	submission.UserID = "verified-subject"
	return submission
}

func TestFileProcessorRejectsDuplicates(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "feedback")
	process, err := newFileProcessor(directory)
	if err != nil {
		t.Fatal(err)
	}
	submission := testSubmission(t)
	if err := process(context.Background(), submission); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, submission.RequestID+".json")
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	submission.Feedback.Comment = "Must not replace the first submission"
	if err := process(context.Background(), submission); !errors.Is(err, ErrFeedbackExists) {
		t.Fatalf("duplicate error = %v", err)
	}
	saved, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(original, saved) {
		t.Fatal("duplicate changed original file")
	}
}

func TestFileProcessorConcurrentCreate(t *testing.T) {
	directory := t.TempDir()
	process, err := newFileProcessor(directory)
	if err != nil {
		t.Fatal(err)
	}
	submission := testSubmission(t)
	const writers = 10
	results := make(chan error, writers)
	var wg sync.WaitGroup
	for range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results <- process(context.Background(), submission)
		}()
	}
	wg.Wait()
	close(results)
	successes, duplicates := 0, 0
	for err := range results {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, ErrFeedbackExists):
			duplicates++
		default:
			t.Fatal(err)
		}
	}
	if successes != 1 || duplicates != writers-1 {
		t.Fatalf("successes=%d duplicates=%d", successes, duplicates)
	}
	saved, err := os.ReadFile(filepath.Join(directory, submission.RequestID+".json"))
	if err != nil || !json.Valid(saved) {
		t.Fatalf("invalid saved JSON: err=%v data=%s", err, saved)
	}
}

func TestFileProcessorErrors(t *testing.T) {
	for _, directory := range []string{"", "relative/path"} {
		if _, err := newFileProcessor(directory); err == nil {
			t.Fatalf("accepted directory %q", directory)
		}
	}
	root := t.TempDir()
	notDirectory := filepath.Join(root, "file")
	if err := os.WriteFile(notDirectory, []byte("existing"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := newFileProcessor(notDirectory); err == nil {
		t.Fatal("accepted a regular file as output directory")
	}
	directory := filepath.Join(root, "feedback")
	process, err := newFileProcessor(directory)
	if err != nil {
		t.Fatal(err)
	}
	submission := testSubmission(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := process(ctx, submission); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel error = %v", err)
	}
	submission.RequestID = "../outside"
	if err := process(context.Background(), submission); err == nil {
		t.Fatal("accepted unsafe filename")
	}
	files, err := os.ReadDir(directory)
	if err != nil || len(files) != 0 {
		t.Fatalf("failed submissions created files: %v, %v", files, err)
	}
	if err := os.Remove(directory); err != nil {
		t.Fatal(err)
	}
	submission = testSubmission(t)
	if err := process(context.Background(), submission); err == nil {
		t.Fatal("write to missing directory returned success")
	}
}

func TestFileProcessorDoesNotFollowExistingSymlink(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "feedback")
	process, err := newFileProcessor(directory)
	if err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "outside.json")
	if err := os.WriteFile(target, []byte("untouched"), 0600); err != nil {
		t.Fatal(err)
	}
	submission := testSubmission(t)
	if err := os.Symlink(target, filepath.Join(directory, submission.RequestID+".json")); err != nil {
		t.Fatal(err)
	}
	if err := process(context.Background(), submission); !errors.Is(err, ErrFeedbackExists) {
		t.Fatalf("symlink error = %v", err)
	}
	saved, err := os.ReadFile(target)
	if err != nil || string(saved) != "untouched" {
		t.Fatalf("symlink target changed: %s, %v", saved, err)
	}
}
