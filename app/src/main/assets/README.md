# Model Assets Setup

This folder needs to contain the following files for the nomic-embed-text-v1.5 model:

## Required Files

1. **model.onnx** - The ONNX model file
2. **vocab.txt** - The BERT vocabulary file

## Download Instructions

### Option 1: Using Hugging Face CLI

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the ONNX model
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 onnx/model.onnx --local-dir ./temp
cp ./temp/onnx/model.onnx app/src/main/assets/

# Download vocab.txt (from bert-base-uncased which nomic uses)
huggingface-cli download bert-base-uncased vocab.txt --local-dir ./temp
cp ./temp/vocab.txt app/src/main/assets/
```

### Option 2: Manual Download

1. Go to https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/main/onnx
2. Download `model.onnx` and place it in this folder
3. Go to https://huggingface.co/bert-base-uncased/blob/main/vocab.txt
4. Download `vocab.txt` and place it in this folder

### Option 3: Using the download script

Run the provided download script from the project root:

```bash
./download_model.sh
```

## File Sizes

- model.onnx: ~134 MB
- vocab.txt: ~232 KB

## Note

The model file is large (~134MB). Make sure you have enough storage on your device.
For production apps, consider downloading the model at runtime or using Android App Bundles
with on-demand delivery.
