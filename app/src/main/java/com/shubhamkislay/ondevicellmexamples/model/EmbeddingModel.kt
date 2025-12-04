package com.shubhamkislay.ondevicellmexamples.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.shubhamkislay.ondevicellmexamples.tokenizer.BertTokenizer
import java.io.File
import java.io.FileOutputStream
import java.nio.LongBuffer
import kotlin.math.sqrt

/**
 * Embedding model wrapper for nomic-embed-text-v1.5 ONNX model.
 * Handles model loading, inference, and embedding generation.
 */
class EmbeddingModel(private val context: Context) {
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var tokenizer: BertTokenizer? = null
    
    private var isInitialized = false
    
    val embeddingDimension = 768 // nomic-embed-text-v1.5 produces 768-dim embeddings
    
    /**
     * Initialize the model and tokenizer.
     * Call this before generating embeddings.
     */
    fun initialize(): Result<Unit> {
        return try {
            // Initialize tokenizer
            tokenizer = BertTokenizer(context)
            
            // Initialize ONNX Runtime
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            // Copy model from assets to files dir (avoids loading entire model into memory)
            val modelFile = copyAssetToFile("model.onnx")
            
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnvironment?.createSession(modelFile.absolutePath, sessionOptions)
            
            isInitialized = true
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Copy asset file to app's files directory for memory-efficient loading.
     * Only copies if file doesn't exist or is different size.
     */
    private fun copyAssetToFile(assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        
        // Check if already copied (by file size)
        val assetFd = context.assets.openFd(assetName)
        val assetSize = assetFd.length
        assetFd.close()
        
        if (outFile.exists() && outFile.length() == assetSize) {
            return outFile
        }
        
        // Copy with buffered streaming to avoid OOM
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
     * Generate embedding for the given text.
     * For nomic-embed-text, prepend "search_query: " for queries or "search_document: " for documents.
     */
    fun generateEmbedding(text: String, isQuery: Boolean = true): Result<FloatArray> {
        if (!isInitialized) {
            return Result.failure(IllegalStateException("Model not initialized. Call initialize() first."))
        }
        
        val env = ortEnvironment ?: return Result.failure(IllegalStateException("ORT environment is null"))
        val session = ortSession ?: return Result.failure(IllegalStateException("ORT session is null"))
        val tok = tokenizer ?: return Result.failure(IllegalStateException("Tokenizer is null"))
        
        return try {
            // Prepend task prefix as per nomic-embed-text documentation
            val prefixedText = if (isQuery) "search_query: $text" else "search_document: $text"
            
            // Tokenize
            val encoded = tok.encode(prefixedText, maxLength = 512)
            
            // Create input tensors
            val batchSize = 1L
            val seqLength = encoded.inputIds.size.toLong()
            val shape = longArrayOf(batchSize, seqLength)
            
            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(encoded.inputIds),
                shape
            )
            
            val attentionMaskTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(encoded.attentionMask),
                shape
            )
            
            val tokenTypeIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(encoded.tokenTypeIds),
                shape
            )
            
            // Run inference
            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attentionMaskTensor,
                "token_type_ids" to tokenTypeIdsTensor
            )
            
            val results = session.run(inputs)
            
            // Get output - the model outputs last_hidden_state
            // We need to do mean pooling over the sequence dimension
            val outputTensor = results[0].value as Array<Array<FloatArray>>
            
            // Mean pooling with attention mask
            val embedding = meanPooling(outputTensor[0], encoded.attentionMask)
            
            // Normalize the embedding (L2 normalization)
            val normalizedEmbedding = l2Normalize(embedding)
            
            // Clean up tensors
            inputIdsTensor.close()
            attentionMaskTensor.close()
            tokenTypeIdsTensor.close()
            results.close()
            
            Result.success(normalizedEmbedding)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * Mean pooling over token embeddings, weighted by attention mask.
     */
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
        
        // Average
        if (validTokenCount > 0) {
            for (j in 0 until embeddingDim) {
                result[j] /= validTokenCount
            }
        }
        
        return result
    }
    
    /**
     * L2 normalize the embedding vector.
     */
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
    
    /**
     * Release resources.
     */
    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        tokenizer = null
        isInitialized = false
    }
}
