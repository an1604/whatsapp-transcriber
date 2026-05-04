#!/bin/sh
set -e

OLLAMA_HOST="${OLLAMA_HOST:-ollama}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
MAX_WAIT="${OLLAMA_MAX_WAIT:-60}"

elapsed=0
until curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/version" > /dev/null 2>&1; do
    if [ "$elapsed" -ge "$MAX_WAIT" ]; then
        echo "Ollama did not become ready within ${MAX_WAIT}s — aborting." >&2
        exit 1
    fi
    echo "Waiting for Ollama at ${OLLAMA_HOST}:${OLLAMA_PORT}… (${elapsed}s elapsed)"
    sleep 2
    elapsed=$((elapsed + 2))
done

echo "Ollama is ready."
exec "$@"
