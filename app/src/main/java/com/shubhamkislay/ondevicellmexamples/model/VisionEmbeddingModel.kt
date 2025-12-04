package com.shubhamkislay.ondevicellmexamples.model

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

/**
 * Vision embedding model wrapper for nomic-embed-vision-v1.5 ONNX model.
 * Generates embeddings from images that are compatible with nomic-embed-text embeddings.
 */
class VisionEmbeddingModel(private val context: Context) {
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isInitialized = false
    
    val embeddingDimension = 768
    
    // Image preprocessing constants (ImageNet normalization)
    private val imageMean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val imageStd = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val imageSize = 224 // nomic-embed-vision uses 224x224
    
    /**
     * Initialize the model. Call this before generating embeddings.
     */
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
     * Generate embedding from a Bitmap.
     */
    fun generateEmbedding(bitmap: Bitmap): Result<FloatArray> {
        if (!isInitialized) {
            return Result.failure(IllegalStateException("Model not initialized"))
        }
        
        val env = ortEnvironment ?: return Result.failure(IllegalStateException("ORT environment is null"))
        val session = ortSession ?: return Result.failure(IllegalStateException("ORT session is null"))
        
        return try {
            // Preprocess image
            val inputTensor = preprocessImage(bitmap, env)
            
            val inputs = mapOf("pixel_values" to inputTensor)
            val results = session.run(inputs)
            
            // Get output embedding - handle different output shapes
            val output = results[0].value
            
            val embedding: FloatArray = when (output) {
                is FloatArray -> output
                is Array<*> -> {
                    when (val first = output[0]) {
                        is FloatArray -> first // Shape: [1, 768]
                        is Array<*> -> {
                            // Shape: [1, seq_len, 768] - take first token (CLS)
                            @Suppress("UNCHECKED_CAST")
                            (first as Array<FloatArray>)[0]
                        }
                        else -> throw IllegalStateException("Unexpected inner type: ${first?.javaClass}")
                    }
                }
                else -> throw IllegalStateException("Unexpected output type: ${output?.javaClass}")
            }
            
            // L2 normalize
            val normalizedEmbedding = l2Normalize(embedding)
            
            inputTensor.close()
            results.close()
            
            Result.success(normalizedEmbedding)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Generate embedding from image URI.
     */
    fun generateEmbedding(imageUri: Uri): Result<FloatArray> {
        return try {
            val inputStream = context.contentResolver.openInputStream(imageUri)
                ?: return Result.failure(IllegalArgumentException("Cannot open image URI"))
            
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            
            if (bitmap == null) {
                return Result.failure(IllegalArgumentException("Cannot decode image"))
            }
            
            val result = generateEmbedding(bitmap)
            bitmap.recycle()
            result
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Preprocess image for the vision model.
     * Resize to 384x384, normalize with ImageNet stats, convert to CHW format.
     */
    private fun preprocessImage(bitmap: Bitmap, env: OrtEnvironment): OnnxTensor {
        // Resize to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        
        // Convert to float array in CHW format with normalization
        val pixels = IntArray(imageSize * imageSize)
        resizedBitmap.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        
        val floatValues = FloatArray(3 * imageSize * imageSize)
        
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // CHW format with normalization
            floatValues[i] = (r - imageMean[0]) / imageStd[0]                           // R channel
            floatValues[imageSize * imageSize + i] = (g - imageMean[1]) / imageStd[1]   // G channel
            floatValues[2 * imageSize * imageSize + i] = (b - imageMean[2]) / imageStd[2] // B channel
        }
        
        if (resizedBitmap != bitmap) {
            resizedBitmap.recycle()
        }
        
        val shape = longArrayOf(1, 3, imageSize.toLong(), imageSize.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floatValues), shape)
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
        isInitialized = false
    }
}
