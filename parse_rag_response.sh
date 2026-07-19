#!/bin/bash

# Parse RAG API response and format for readability
# Usage: ./parse_rag_response.sh <json_file> or cat response.json | ./parse_rag_response.sh

# Read input from file or stdin
INPUT="${1:-/dev/stdin}"

# Extract and parse the JSON response
if command -v jq &> /dev/null; then
    # Use jq if available (cleaner)
    jq -r '.answer' "$INPUT" | sed 's/\\n/\n/g'
else
    # Fallback to grep/sed for systems without jq
    grep -o '"answer":"[^"]*' "$INPUT" \
        | sed 's/"answer":"//' \
        | sed 's/\\n/\n/g'
fi

# Also show sources if available
echo ""
echo "=== Sources ==="
if command -v jq &> /dev/null; then
    jq '.sources // "No sources metadata"' "$INPUT" 2>/dev/null || echo "No sources metadata"
else
    echo "(Install jq to see sources metadata)"
fi
