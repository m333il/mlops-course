#!/bin/bash
set -e

MODEL_NAME="fashion_classifier"
VERSION="1.0"
MODEL_FILE="serve/model.pt"
HANDLER_FILE="serve/handler.py"
EXPORT_PATH="model-store"

mkdir -p "$EXPORT_PATH"

rm -f "$EXPORT_PATH/$MODEL_NAME.mar"

torch-model-archiver \
    --model-name "$MODEL_NAME" \
    --version "$VERSION" \
    --serialized-file "$MODEL_FILE" \
    --handler "$HANDLER_FILE" \
    --export-path "$EXPORT_PATH" \
    --force
