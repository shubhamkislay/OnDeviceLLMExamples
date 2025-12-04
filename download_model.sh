#!/bin/bash

# Download script for nomic-embed-text-v1.5 ONNX model and vocabulary

ASSETS_DIR="app/src/main/assets"

echo "Creating assets directory..."
mkdir -p "$ASSETS_DIR"

echo ""
echo "Downloading nomic-embed-text-v1.5 ONNX model (quantized INT8 version)..."
echo "This may take a while (~137MB)..."

# Download the quantized ONNX model from Hugging Face (better for mobile)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx" \
    -o "$ASSETS_DIR/model.onnx" \
    --progress-bar

if [ $? -eq 0 ]; then
    echo "✓ Model downloaded successfully"
else
    echo "✗ Failed to download model"
    exit 1
fi

echo ""
echo "Downloading BERT vocabulary..."

# Download vocab.txt from bert-base-uncased (which nomic-embed-text uses)
curl -L "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt" \
    -o "$ASSETS_DIR/vocab.txt" \
    --progress-bar

if [ $? -eq 0 ]; then
    echo "✓ Vocabulary downloaded successfully"
else
    echo "✗ Failed to download vocabulary"
    exit 1
fi

echo ""
echo "Verifying downloads..."
echo ""

if [ -f "$ASSETS_DIR/model.onnx" ]; then
    MODEL_SIZE=$(ls -lh "$ASSETS_DIR/model.onnx" | awk '{print $5}')
    echo "✓ model.onnx: $MODEL_SIZE"
else
    echo "✗ model.onnx not found"
fi

if [ -f "$ASSETS_DIR/vocab.txt" ]; then
    VOCAB_SIZE=$(ls -lh "$ASSETS_DIR/vocab.txt" | awk '{print $5}')
    VOCAB_LINES=$(wc -l < "$ASSETS_DIR/vocab.txt")
    echo "✓ vocab.txt: $VOCAB_SIZE ($VOCAB_LINES tokens)"
else
    echo "✗ vocab.txt not found"
fi

echo ""
echo "Setup complete! You can now build and run the app."
