package com.shubhamkislay.ondevicellmexamples

import com.shubhamkislay.ondevicellmexamples.viewmodel.EmbeddingUiState
import org.junit.Assert.*
import org.junit.Test

/**
 * Unit tests for EmbeddingUiState data class
 */
class EmbeddingUiStateTest {
    
    @Test
    fun `default state has correct initial values`() {
        val state = EmbeddingUiState()
        
        assertTrue(state.isModelLoading)
        assertFalse(state.isGenerating)
        assertNull(state.modelLoadError)
        assertEquals("", state.inputText)
        assertNull(state.embedding)
        assertNull(state.error)
        assertEquals(0L, state.inferenceTimeMs)
    }
    
    @Test
    fun `state copy works correctly`() {
        val state = EmbeddingUiState()
        val newState = state.copy(
            isModelLoading = false,
            inputText = "test input"
        )
        
        assertFalse(newState.isModelLoading)
        assertEquals("test input", newState.inputText)
        // Other values should remain default
        assertFalse(newState.isGenerating)
        assertNull(newState.embedding)
    }
    
    @Test
    fun `embedding array is stored correctly`() {
        val embedding = FloatArray(768) { it.toFloat() / 768f }
        val state = EmbeddingUiState(embedding = embedding)
        
        assertNotNull(state.embedding)
        assertEquals(768, state.embedding?.size)
        assertEquals(0f, state.embedding?.get(0) ?: -1f, 0.001f)
    }
}
