# On-Device Embeddings with Nomic Embed v1.5

Run [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) and [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) ONNX models locally on Android - completely offline, no API calls needed.

## Features

- 🔒 100% offline - no network required after setup
- ⚡ Fast inference using ONNX Runtime
- 📊 768-dimensional embeddings
- 🖼️ Multimodal - embed both text AND images
- 🎯 Shared vector space - compare text with images directly
- 📱 Optimized for mobile with quantized models

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/shubhamkislay/OnDeviceLLMExamples.git
cd OnDeviceLLMExamples
```

### 2. Download the model files

The ONNX model files are too large for GitHub (~160MB total), so they must be downloaded separately.

```bash
# Make the script executable (first time only)
chmod +x download_model.sh

# Run the download script
./download_model.sh
```

This will download:
- `model.onnx` - Text embedding model (~106MB)
- `vision_model.onnx` - Image embedding model (~54MB)  
- `vocab.txt` - BERT vocabulary (~226KB)

Files are automatically placed in `app/src/main/assets/`.

### 3. Build and run

```bash
./gradlew assembleDebug
./gradlew installDebug
```

Or open in Android Studio and click Run.

---

## Integration Guide (Add to Your Own Project)

Want to add embedding functionality to your existing Android app? Follow these steps.

### Step 1: Add Dependencies

Add to `gradle/libs.versions.toml`:

```toml
[versions]
onnxruntime = "1.19.2"
coroutines = "1.8.1"
gson = "2.11.0"
coil = "2.6.0"

[libraries]
onnxruntime-android = { group = "com.microsoft.onnxruntime", name = "onnxruntime-android", version.ref = "onnxruntime" }
kotlinx-coroutines-android = { group = "org.jetbrains.kotlinx", name = "kotlinx-coroutines-android", version.ref = "coroutines" }
gson = { group = "com.google.code.gson", name = "gson", version.ref = "gson" }
coil-compose = { group = "io.coil-kt", name = "coil-compose", version.ref = "coil" }
```

Add to `app/build.gradle.kts`:

```kotlin
android {
    androidResources {
        noCompress += listOf("onnx")
    }
}

dependencies {
    implementation(libs.onnxruntime.android)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.gson)
    implementation(libs.coil.compose) // For image loading
}
```

### Step 2: Download Model Files

The model files are too large for GitHub (~160MB total), so you need to download them separately.

#### Required Files

| File | Size | Purpose |
|------|------|---------|
| `model.onnx` | ~106MB | Text embedding model |
| `vision_model.onnx` | ~54MB | Image embedding model |
| `vocab.txt` | ~226KB | BERT tokenizer vocabulary |

These files must be placed in: `app/src/main/assets/`

#### Option A: Copy the Download Script (Recommended)

1. Copy `download_model.sh` from this repo to your project root
2. Run it:

```bash
# Make executable (first time only)
chmod +x download_model.sh

# Run from your project root directory
./download_model.sh
```

The script will:
- Create `app/src/main/assets/` if it doesn't exist
- Download all 3 required files from Hugging Face
- Show progress and verify downloads

#### Option B: Manual Download via Terminal

```bash
# Create assets directory
mkdir -p app/src/main/assets

# Download text model (~106MB)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_q4f16.onnx" \
    -o "app/src/main/assets/model.onnx"

# Download vision model (~54MB)
curl -L "https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/main/onnx/model_bnb4.onnx" \
    -o "app/src/main/assets/vision_model.onnx"

# Download BERT vocabulary (~226KB)
curl -L "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt" \
    -o "app/src/main/assets/vocab.txt"
```

#### Option C: Manual Download via Browser

1. **Text Model**: 
   - Go to: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/main/onnx
   - Click on `model_q4f16.onnx` → Download
   - Rename to `model.onnx` and place in `app/src/main/assets/`

2. **Vision Model**:
   - Go to: https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/tree/main/onnx
   - Click on `model_bnb4.onnx` → Download
   - Rename to `vision_model.onnx` and place in `app/src/main/assets/`

3. **Vocabulary**:
   - Go to: https://huggingface.co/bert-base-uncased/blob/main/vocab.txt
   - Click Download
   - Place in `app/src/main/assets/`

#### Verify Your Setup

Your `app/src/main/assets/` folder should look like this:

```
app/src/main/assets/
├── model.onnx          (~106 MB)
├── vision_model.onnx   (~54 MB)
└── vocab.txt           (~226 KB)
```

You can verify with:
```bash
ls -lh app/src/main/assets/
```

#### Add to .gitignore (Important!)

The model files are too large for GitHub (100MB limit). Add these lines to your `.gitignore` to prevent accidentally committing them:

```gitignore
# ONNX model files (too large for GitHub - download separately)
app/src/main/assets/model.onnx
app/src/main/assets/vision_model.onnx
```

> **Note**: `vocab.txt` is small (~226KB) and can be committed to git if you prefer.

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

### Step 4: Add the Text Embedding Model

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

    fun initialize(): Result<Unit> {
        return try {
            tokenizer = BertTokenizer(context)
            ortEnvironment = OrtEnvironment.getEnvironment()
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
        
        if (outFile.exists() && outFile.length() == assetSize) return outFile
        
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

    fun generateEmbedding(text: String, isQuery: Boolean = true): Result<FloatArray> {
        if (!isInitialized) return Result.failure(IllegalStateException("Model not initialized"))
        
        val env = ortEnvironment ?: return Result.failure(IllegalStateException("ORT environment is null"))
        val session = ortSession ?: return Result.failure(IllegalStateException("ORT session is null"))
        val tok = tokenizer ?: return Result.failure(IllegalStateException("Tokenizer is null"))
        
        return try {
            val prefixedText = if (isQuery) "search_query: $text" else "search_document: $text"
            val encoded = tok.encode(prefixedText, maxLength = 512)
            
            val shape = longArrayOf(1L, encoded.inputIds.size.toLong())
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
        for (value in embedding) sumSquares += value * value
        val norm = sqrt(sumSquares)
        return if (norm > 0) FloatArray(embedding.size) { embedding[it] / norm } else embedding
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


### Step 5: Add the Vision Embedding Model

Create `VisionEmbeddingModel.kt`:

```kotlin
package com.yourpackage.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.sqrt

class VisionEmbeddingModel(private val context: Context) {
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false
    
    val embeddingDimension = 768
    
    // ImageNet normalization
    private val imageMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imageStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val imageSize = 224

    fun initialize(): Result<Unit> {
        return try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val modelFile = copyAssetToFile("vision_model.onnx")
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
        
        if (outFile.exists() && outFile.length() == assetSize) return outFile
        
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

    fun generateEmbedding(bitmap: Bitmap): Result<FloatArray> {
        if (!isInitialized) return Result.failure(IllegalStateException("Model not initialized"))
        
        val env = ortEnvironment ?: return Result.failure(IllegalStateException("ORT environment is null"))
        val session = ortSession ?: return Result.failure(IllegalStateException("ORT session is null"))
        
        return try {
            val inputTensor = preprocessImage(bitmap, env)
            val inputs = mapOf("pixel_values" to inputTensor)
            val results = session.run(inputs)
            
            val output = results[0].value
            val embedding: FloatArray = when (output) {
                is FloatArray -> output
                is Array<*> -> {
                    when (val first = output[0]) {
                        is FloatArray -> first
                        is Array<*> -> {
                            @Suppress("UNCHECKED_CAST")
                            (first as Array<FloatArray>)[0]
                        }
                        else -> throw IllegalStateException("Unexpected output type")
                    }
                }
                else -> throw IllegalStateException("Unexpected output type")
            }
            
            val normalizedEmbedding = l2Normalize(embedding)
            inputTensor.close()
            results.close()
            
            Result.success(normalizedEmbedding)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    fun generateEmbedding(imageUri: Uri): Result<FloatArray> {
        return try {
            val inputStream = context.contentResolver.openInputStream(imageUri)
                ?: return Result.failure(IllegalArgumentException("Cannot open image URI"))
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            if (bitmap == null) return Result.failure(IllegalArgumentException("Cannot decode image"))
            val result = generateEmbedding(bitmap)
            bitmap.recycle()
            result
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    private fun preprocessImage(bitmap: Bitmap, env: OrtEnvironment): OnnxTensor {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        val pixels = IntArray(imageSize * imageSize)
        resizedBitmap.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        
        val floatValues = FloatArray(3 * imageSize * imageSize)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // CHW format with normalization
            floatValues[i] = (r - imageMean[0]) / imageStd[0]
            floatValues[imageSize * imageSize + i] = (g - imageMean[1]) / imageStd[1]
            floatValues[2 * imageSize * imageSize + i] = (b - imageMean[2]) / imageStd[2]
        }
        
        if (resizedBitmap != bitmap) resizedBitmap.recycle()
        
        val shape = longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floatValues), shape)
    }
    
    private fun l2Normalize(embedding: FloatArray): FloatArray {
        var sumSquares = 0f
        for (value in embedding) sumSquares += value * value
        val norm = sqrt(sumSquares)
        return if (norm > 0) FloatArray(embedding.size) { embedding[it] / norm } else embedding
    }

    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        isInitialized = false
    }
}
```

### Step 6: Use in Your App

#### ViewModel with Both Models

```kotlin
class EmbeddingViewModel(application: Application) : AndroidViewModel(application) {
    
    private val textModel = EmbeddingModel(application)
    private val visionModel = VisionEmbeddingModel(application)
    
    private val _textEmbedding = MutableStateFlow<FloatArray?>(null)
    val textEmbedding: StateFlow<FloatArray?> = _textEmbedding.asStateFlow()
    
    private val _imageEmbedding = MutableStateFlow<FloatArray?>(null)
    val imageEmbedding: StateFlow<FloatArray?> = _imageEmbedding.asStateFlow()
    
    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                textModel.initialize()
                visionModel.initialize()
            }
        }
    }
    
    fun generateTextEmbedding(text: String) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.Default) {
                textModel.generateEmbedding(text)
            }
            _textEmbedding.value = result.getOrNull()
        }
    }
    
    fun generateImageEmbedding(uri: Uri) {
        viewModelScope.launch {
            val result = withContext(Dispatchers.Default) {
                visionModel.generateEmbedding(uri)
            }
            _imageEmbedding.value = result.getOrNull()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        textModel.close()
        visionModel.close()
    }
}
```

#### Cross-Modal Similarity (Text ↔ Image)

```kotlin
fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
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

// Compare text query with image
val textEmb = textModel.generateEmbedding("a photo of a cat").getOrNull()
val imageEmb = visionModel.generateEmbedding(catImageUri).getOrNull()

if (textEmb != null && imageEmb != null) {
    val similarity = cosineSimilarity(textEmb, imageEmb)
    println("Text-Image similarity: $similarity")
}
```

---

## Model Variants

### Text Models (nomic-embed-text-v1.5)

| Model | Size | Notes |
|-------|------|-------|
| `model_q4f16.onnx` | 106MB | Recommended |
| `model_quantized.onnx` | 131MB | INT8 |
| `model_fp16.onnx` | 261MB | Higher quality |

### Vision Models (nomic-embed-vision-v1.5)

| Model | Size | Notes |
|-------|------|-------|
| `model_bnb4.onnx` | 54MB | Recommended |
| `model_int8.onnx` | 92MB | INT8 |
| `model_fp16.onnx` | 179MB | Higher quality |

---

## Important Notes

### Shared Vector Space
Text and image embeddings exist in the same 768-dimensional space. You can directly compare:
- Text ↔ Text
- Image ↔ Image  
- Text ↔ Image (cross-modal search!)

### Task Prefixes (Text Only)
- `"search_query: "` for queries
- `"search_document: "` for documents

### Image Preprocessing
- Images are resized to 224x224
- Normalized with ImageNet mean/std
- Converted to CHW format

### Threading
- `initialize()` → `Dispatchers.IO`
- `generateEmbedding()` → `Dispatchers.Default`

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Text Embedding                       │
├─────────────────────────────────────────────────────────┤
│  Text → Tokenizer → ONNX Model → Mean Pool → L2 Norm   │
└─────────────────────────────────────────────────────────┘
                           ↓
                    768-dim Vector ←──── Shared Space
                           ↑
┌─────────────────────────────────────────────────────────┐
│                   Image Embedding                       │
├─────────────────────────────────────────────────────────┤
│  Image → Resize/Norm → ONNX Model → L2 Norm            │
└─────────────────────────────────────────────────────────┘
```

---

## License

MIT License

## Credits

- [Nomic AI](https://www.nomic.ai/) for nomic-embed-text and nomic-embed-vision models
- [ONNX Runtime](https://onnxruntime.ai/) for the inference engine
- [Hugging Face](https://huggingface.co/) for model hosting
