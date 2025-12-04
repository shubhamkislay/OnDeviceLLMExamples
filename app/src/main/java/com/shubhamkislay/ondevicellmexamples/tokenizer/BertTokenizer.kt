package com.shubhamkislay.ondevicellmexamples.tokenizer

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.InputStreamReader

/**
 * BERT WordPiece tokenizer for nomic-embed-text-v1.5 model.
 * Handles text tokenization compatible with the ONNX model's expected input format.
 */
class BertTokenizer(context: Context) {
    
    private val vocab: Map<String, Int>
    private val idToToken: Map<Int, String>
    
    // Special tokens
    private val clsToken = "[CLS]"
    private val sepToken = "[SEP]"
    private val padToken = "[PAD]"
    private val unkToken = "[UNK]"
    
    val clsTokenId: Int
    val sepTokenId: Int
    val padTokenId: Int
    private val unkTokenId: Int
    
    val maxLength = 512
    
    init {
        // Load vocabulary from assets
        val inputStream = context.assets.open("vocab.txt")
        val reader = InputStreamReader(inputStream)
        val lines = reader.readLines()
        reader.close()
        
        vocab = lines.mapIndexed { index, token -> token to index }.toMap()
        idToToken = vocab.entries.associate { it.value to it.key }
        
        clsTokenId = vocab[clsToken] ?: 101
        sepTokenId = vocab[sepToken] ?: 102
        padTokenId = vocab[padToken] ?: 0
        unkTokenId = vocab[unkToken] ?: 100
    }

    /**
     * Tokenize input text and return token IDs, attention mask, and token type IDs.
     */
    fun encode(text: String, maxLength: Int = this.maxLength): TokenizerOutput {
        // Basic preprocessing
        val cleanedText = text.lowercase().trim()
        
        // Basic tokenization (split on whitespace and punctuation)
        val tokens = basicTokenize(cleanedText)
        
        // WordPiece tokenization
        val wordPieceTokens = mutableListOf<String>()
        for (token in tokens) {
            wordPieceTokens.addAll(wordPieceTokenize(token))
        }
        
        // Truncate if necessary (leave room for [CLS] and [SEP])
        val maxTokens = maxLength - 2
        val truncatedTokens = if (wordPieceTokens.size > maxTokens) {
            wordPieceTokens.take(maxTokens)
        } else {
            wordPieceTokens
        }
        
        // Build final token sequence: [CLS] + tokens + [SEP]
        val finalTokens = mutableListOf(clsToken)
        finalTokens.addAll(truncatedTokens)
        finalTokens.add(sepToken)
        
        // Convert to IDs
        val inputIds = finalTokens.map { (vocab[it] ?: unkTokenId).toLong() }.toLongArray()
        
        // Create attention mask (1 for real tokens, 0 for padding)
        val attentionMask = LongArray(inputIds.size) { 1L }
        
        // Token type IDs (all 0 for single sequence)
        val tokenTypeIds = LongArray(inputIds.size) { 0L }
        
        // Pad to maxLength if needed
        val paddedInputIds = padArray(inputIds, maxLength, padTokenId.toLong())
        val paddedAttentionMask = padArray(attentionMask, maxLength, 0L)
        val paddedTokenTypeIds = padArray(tokenTypeIds, maxLength, 0L)
        
        return TokenizerOutput(
            inputIds = paddedInputIds,
            attentionMask = paddedAttentionMask,
            tokenTypeIds = paddedTokenTypeIds
        )
    }
    
    private fun basicTokenize(text: String): List<String> {
        // Split on whitespace and punctuation, keeping punctuation as separate tokens
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
        
        if (currentToken.isNotEmpty()) {
            tokens.add(currentToken.toString())
        }
        
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
                // Character not in vocab, use [UNK]
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
        // ASCII punctuation
        if ((cp in 33..47) || (cp in 58..64) || (cp in 91..96) || (cp in 123..126)) {
            return true
        }
        return Character.getType(char) == Character.OTHER_PUNCTUATION.toInt()
    }
    
    private fun padArray(array: LongArray, targetLength: Int, padValue: Long): LongArray {
        if (array.size >= targetLength) return array.copyOf(targetLength)
        return LongArray(targetLength) { i ->
            if (i < array.size) array[i] else padValue
        }
    }
}

data class TokenizerOutput(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val tokenTypeIds: LongArray
)
