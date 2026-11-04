package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

var ErrFeedbackExists = errors.New("feedback already exists")

func newFileProcessor(directory string) (FeedbackProcessor, error) {
	if !filepath.IsAbs(directory) {
		return nil, errors.New("FEEDBACK_OUTPUT_DIR must be an absolute directory path")
	}
	if err := os.MkdirAll(directory, 0750); err != nil {
		return nil, fmt.Errorf("create feedback output directory: %w", err)
	}
	return func(ctx context.Context, submission FeedbackSubmission) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := validateRequestID(submission.RequestID); err != nil {
			return err
		}
		data, err := json.MarshalIndent(submission, "", "  ")
		if err != nil {
			return fmt.Errorf("encode feedback: %w", err)
		}
		path := filepath.Join(directory, submission.RequestID+".json")
		// Do not truncate an existing submission, including a concurrent writer's file.
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if errors.Is(err, os.ErrExist) {
			return ErrFeedbackExists
		}
		if err != nil {
			return fmt.Errorf("create feedback file: %w", err)
		}
		_, writeErr := file.Write(append(data, '\n'))
		// On GCS FUSE, closing the file finalizes the upload; check this before acknowledging.
		closeErr := file.Close()
		if err := errors.Join(writeErr, closeErr); err != nil {
			// A failed FUSE close has an uncertain remote outcome; do not delete a
			// potentially committed object. Leave recovery to the operator.
			return fmt.Errorf("write feedback file (inspect file before retrying): %w", err)
		}
		return nil
	}, nil
}
