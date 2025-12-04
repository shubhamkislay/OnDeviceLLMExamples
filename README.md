# On-Device Text Embeddings with Nomic Embed Text v1.5

Run the [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) ONNX model locally on Android to generate text embeddings - completely offline, no API calls needed.

## Features

- 🔒 100% offline - no network required after setup
- ⚡ Fast inference using ONNX Runtime
- 📊 768-dimensional embeddings
- 🎯 Perfect for semantic search, RAG, similarity matching
- 📱 Optimized for mobile with quantized model

## Quick Start (This Project)

### 1. Clone and download model

```bash
git clone <repository-url>
cd OnDeviceLLMExamples
./download_model.sh
```

### 2. Build and run

```bash
./gradlew assembleDebug
# Install on connected device
./gradlew installDebug
```

---

## Integration Guide

Follow these steps to add text embedding capabilities to your own Android project.

### Step 1: Add Dependencies

Add to your `gradle/libs.versions.toml`:

```toml
[versions]
onnxruntime = "1.19.2"
coroutines = "1.8.1"
gson = "2.11.0"

[libraries]
onnxruntime-android = { group = "com.microsoft.onnxruntime", name = "onnxruntime-android", version.ref = "onnxruntime" }
kotlinx-coroutines-android = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }
gson = { group = "com.google.code.gson", name = "gson", version.ref = "gson" }
```

Add to your `app/build.gradle.kts`:

```kotlin
android {
    // Prevent compression of ONNX model files
    androidResources {
        noCompress += listOf("onnx")
    }
}

dependencies {
    implementation(libs.onnxruntime.android)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.gson)
}
```

### Step 2: Download Model Files

Create `app/src/main/assets/` folder and download these files:

```bash
# Create assets directory
mkdir -p app/src/main/assets

# Download quantized ONNX model (~106MB)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_q4f16.onnx" \
    -o "app/src/main/assets/model.onnx"

# Download BERT vocabulary
curl -L "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt" \
    -o "app/src/main/assets/vocab.txt"
```

### Step 3: Add the Tokenizer

Create `BertTokenizer.kt`:

```kotlin
package com.yourpackage.tokenizer

import android.content.Context
import java.io.InputStreamReader

class BertTokenizer(context: Context) {
    
    private val vocab: Map<String, Int>
    private val clsToken = "[CLS]"
    private val sepToken = "[SEP]"
    private val padToken = "[PAD]"
    private val unkToken = "[UNK]"
    
    val padTokenId: Int
    private val unkTokenId: Int
    val maxLength = 512
    
    init {
        val inputStream = context.assets.open("vocab.txt")
        val reader = InputStreamReader(inputStream)
        val lines = reader.readLines()
        reader.close()
        
        vocab = lines.mapIndexed { index, token -> token to index }.toMap()
        padTokenId = vocab[padToken] ?: 0
        unkTokenId = vocab[unkToken] ?: 100
    }

    fun encode(text: String, maxLength: Int = this.maxLength): TokenizerOutput {
        val cleanedText = text.lowercase().trim()
        val tokens = basicTokenize(cleanedText)
        
        val wordPieceTokens = mutableListOf<String>()
        for (token in tokens) {
            wordPieceTokens.addAll(wordPieceTokenize(token))
        }
        
        val maxTokens = maxLength - 2
        val truncatedTokens = if (wordPieceTokens.size > maxTokens) {
            wordPieceTokens.take(maxTokens)
        } else {
            wordPieceTokens
        }
        
        val finalTokens = mutableListOf(clsToken)
        finalTokens.addAll(truncatedTokens)
        finalTokens.add(sepToken)
        
        val inputIds = finalTokens.map { (vocab[it] ?: unkTokenId).toLong() }.toLongArray()
        val attentionMask = LongArray(inputIds.size) { 1L }
        val tokenTypeIds = LongArray(inputIds.size) { 0L }
        
        return TokenizerOutput(
            inputIds = padArray(inputIds, maxLength, padTokenId.toLong()),
            attentionMask = padArray(attentionMask, maxLength, 0L),
            tokenTypeIds = padArray(tokenTypeIds, maxLength, 0L)
        )
    }
    
    private fun basicTokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val currentToken = StringBuilder()
        
        for (char in text) {
            when {
                char.isWhitespace() -> {
                    if (currentToken.isNotEmpty()) {
                        tokens.add(currentToken.toString())
                        currentToken.clear()
                    }
                }
                isPunctuation(char) -> {
                    if (currentToken.isNotEmpty()) {
                        tokens.add(currentToken.toString())
                        currentToken.clear()
                    }
                    tokens.add(char.toString())
                }
                else -> currentToken.append(char)
            }
        }
        if (currentToken.isNotEmpty()) tokens.add(currentToken.toString())
        return tokens
    }
    
    private fun wordPieceTokenize(token: String): List<String> {
        if (token.isEmpty()) return emptyList()
        
        val subTokens = mutableListOf<String>()
        var start = 0
        
        while (start < token.length) {
            var end = token.length
            var found = false
            
            while (start < end) {
                val substr = if (start > 0) "##${token.substring(start, end)}" else token.substring(start, end)
                if (vocab.containsKey(substr)) {
                    subTokens.add(substr)
                    found = true
                    break
                }
                end--
            }
            
            if (!found) {
                subTokens.add(unkToken)
                start++
            } else {
                start = end
            }
        }
        return subTokens
    }
    
    private fun isPunctuation(char: Char): Boolean {
        val cp = char.code
        if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)) return true
        return Character.getType(char) == Character.OTHER_PUNCTUATION.toInt()
    }
    
    private fun padArray(array: LongArray, targetLength: Int, padValue: Long): LongArray {
        if (array.size >= targetLength) return array.copyOf(targetLength)
        return LongArray(targetLength) { i -> if (i < array.size) array[i] else padValue }
    }
}

data class TokenizerOutput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val tokenTypeIds: LongArray
)
```


### Step 4: Add the Embedding Model

Create `EmbeddingModel.kt`:

```kotlin
package com.yourpackage.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.yourpackage.tokenizer.BertTokenizer
import java.io.File
import java.io.FileOutputStream
import java.nio.LongBuffer
import kotlin.math.sqrt

class EmbeddingModel(private val context: Context) {
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var tokenizer: BertTokenizer? = null
    private var isInitialized = false
    
    val embeddingDimension = 768

    /**
     * Initialize the model. Call this before generating embeddings.
     * This is a heavy operation - do it once, ideally in a background thread.
     */
    fun initialize(): Result<Unit> {
        return try {
            tokenizer = BertTokenizer(context)
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Copy model to files dir for memory-efficient loading
            val modelFile = copyAssetToFile("model.onnx")
            
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)
            
            isInitialized = true
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    private fun copyAssetToFile(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        
        val assetFd = context.assets.openFd(assetName)
        val assetSize = assetFd.length
        assetFd.close()
        
        if (outFile.exists() && outFile.length() == assetSize) {
            return outFile
        }
        
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output ->
                val buffer = ByteArray(8192)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                }
            }
        }
        return outFile
    }

    /**
     * Generate embedding for text.
     * @param text The input text to embed
     * @param isQuery true for search queries, false for documents being indexed
     * @return 768-dimensional normalized embedding vector
     */
    fun generateEmbedding(text: String, isQuery: Boolean = true): Result<FloatArray> {
        if (!isInitialized) {
            return Result.failure(IllegalStateException("Model not initialized"))
        }
        
        val env = ortEnvironment ?: return Result.failure(IllegalStateException("ORT environment is null"))
        val session = ortSession ?: return Result.failure(IllegalStateException("ORT session is null"))
        val tok = tokenizer ?: return Result.failure(IllegalStateException("Tokenizer is null"))
        
        return try {
            // Nomic requires task prefixes
            val prefixedText = if (isQuery) "search_query: $text" else "search_document: $text"
            
            val encoded = tok.encode(prefixedText, maxLength = 512)
            
            val batchSize = 1L
            val seqLength = encoded.inputIds.size.toLong()
            val shape = longArrayOf(batchSize, seqLength)
            
            val inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.inputIds), shape)
            val attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.attentionMask), shape)
            val tokenTypeIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.tokenTypeIds), shape)
            
            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor,
                "token_type_ids" to tokenTypeIdsTensor
            )
            
            val results = session.run(inputs)
            
            @Suppress("UNCHECKED_CAST")
            val outputTensor = results[0].value as Array<Array<FloatArray>>
            
            val embedding = meanPooling(outputTensor[0], encoded.attentionMask)
            val normalizedEmbedding = l2Normalize(embedding)
            
            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor.close()
            results.close()
            
            Result.success(normalizedEmbedding)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    private fun meanPooling(tokenEmbeddings: Array<FloatArray>, attentionMask: LongArray): FloatArray {
        val embeddingDim = tokenEmbeddings[0].size
        val result = FloatArray(embeddingDim)
        var validTokenCount = 0f
        
        for (i in tokenEmbeddings.indices) {
            if (attentionMask[i] == 1L) {
                for (j in 0 until embeddingDim) {
                    result[j] += tokenEmbeddings[i][j]
                }
                validTokenCount++
            }
        }
        
        if (validTokenCount > 0) {
            for (j in 0 until embeddingDim) {
                result[j] /= validTokenCount
            }
        }
        return result
    }
    
    private fun l2Normalize(embedding: FloatArray): FloatArray {
        var sumSquares = 0f
        for (value in embedding) {
            sumSquares += value * value
        }
        val norm = sqrt(sumSquares)
        
        return if (norm > 0) {
            FloatArray(embedding.size) { embedding[it] / norm }
        } else {
            embedding
        }
    }

    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        tokenizer = null
        isInitialized = false
    }
}
```

### Step 5: Use in Your App

#### Basic Usage

```kotlin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MyRepository(private val context: Context) {
    
    private val embeddingModel = EmbeddingModel(context)
    
    suspend fun initialize() = withContext(Dispatchers.IO) {
        embeddingModel.initialize()
    }
    
    suspend fun getEmbedding(text: String): FloatArray? = withContext(Dispatchers.Default) {
        embeddingModel.generateEmbedding(text).getOrNull()
    }
    
    fun cleanup() {
        embeddingModel.close()
    }
}
```

#### With ViewModel

```kotlin
class EmbeddingViewModel(application: Application) : AndroidViewModel(application) {
    
    private val embeddingModel = EmbeddingModel(application)
    
    private val _embedding = MutableStateFlow<FloatArray?>(null)
    val embedding: StateFlow<FloatArray?> = _embedding.asStateFlow()
    
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                embeddingModel.initialize()
            }
            _isLoading.value = false
        }
    }
    
    fun generateEmbedding(text: String) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.Default) {
                embeddingModel.generateEmbedding(text)
            }
            _embedding.value = result.getOrNull()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        embeddingModel.close()
    }
}
```

#### Compute Similarity Between Texts

```kotlin
fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "Vectors must have same dimension" }
    
    var dotProduct = 0f
    var normA = 0f
    var normB = 0f
    
    for (i in a.indices) {
        dotProduct += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    
    return dotProduct / (sqrt(normA) * sqrt(normB))
}

// Usage
val embedding1 = embeddingModel.generateEmbedding("How do I reset my password?").getOrNull()
val embedding2 = embeddingModel.generateEmbedding("I forgot my login credentials").getOrNull()

if (embedding1 != null && embedding2 != null) {
    val similarity = cosineSimilarity(embedding1, embedding2)
    println("Similarity: $similarity") // ~0.85 (high similarity)
}
```


---

## Model Variants

Different quantized versions are available. Choose based on your needs:

| Model | Size | Quality | Download |
|-------|------|---------|----------|
| `model.onnx` | 522MB | Best | Full precision |
| `model_fp16.onnx` | 261MB | Great | Half precision |
| `model_quantized.onnx` | 131MB | Good | INT8 quantized |
| `model_q4f16.onnx` | 106MB | Good | Q4 + FP16 (recommended) |
| `model_q4.onnx` | 157MB | Good | Q4 quantized |

Download any variant:
```bash
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/<model_name>" \
    -o "app/src/main/assets/model.onnx"
```

---

## Important Notes

### Task Prefixes
Nomic-embed-text requires task prefixes for optimal performance:
- Use `"search_query: "` prefix for queries/questions
- Use `"search_document: "` prefix for documents being indexed

The `EmbeddingModel.generateEmbedding()` handles this automatically via the `isQuery` parameter.

### Memory Considerations
- The model file is copied from assets to internal storage on first run
- This avoids loading the entire model into memory (which causes OOM)
- First launch takes a few extra seconds for the copy
- Subsequent launches are faster

### Threading
- `initialize()` is blocking and should run on `Dispatchers.IO`
- `generateEmbedding()` is CPU-intensive and should run on `Dispatchers.Default`
- Never call these on the main thread

### APK Size
The model adds ~100-130MB to your APK. Consider:
- Using Android App Bundles for on-demand delivery
- Downloading the model at runtime instead of bundling
- Using the smallest quantized model that meets your quality needs

---

## Troubleshooting

### OutOfMemoryError during model loading
Make sure you're using the file-based loading approach (copying to filesDir first), not loading the entire model as a byte array.

### "Not enough space" during installation
The APK is large (~200MB). Free up space on your device/emulator or use a device with more storage.

### Slow first launch
The model is copied from assets to internal storage on first run. This is normal and only happens once.

### Model not found
Ensure `model.onnx` and `vocab.txt` are in `app/src/main/assets/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Your App                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Input     │───▶│  Tokenizer   │───▶│   ONNX    │  │
│  │   Text      │    │  (WordPiece) │    │  Runtime  │  │
│  └─────────────┘    └──────────────┘    └─────┬─────┘  │
│                                               │        │
│  ┌─────────────┐    ┌──────────────┐    ┌─────▼─────┐  │
│  │  Embedding  │◀───│ L2 Normalize │◀───│   Mean    │  │
│  │  (768-dim)  │    │              │    │  Pooling  │  │
│  └─────────────┘    └──────────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## License

MIT License

## Credits

- [Nomic AI](https://www.nomic.ai/) for the nomic-embed-text-v1.5 model
- [ONNX Runtime](https://onnxruntime.ai/) for the inference engine
- [Hugging Face](https://huggingface.co/) for model hosting
