package com.shubhamkislay.ondevicellmexamples.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.shubhamkislay.ondevicellmexamples.model.EmbeddingModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class EmbeddingUiState(
    val isModelLoading: Boolean = true,
    val isGenerating: Boolean = false,
    val modelLoadError: String? = null,
    val inputText: String = "",
    val embedding: FloatArray? = null,
    val error: String? = null,
    val inferenceTimeMs: Long = 0
)

class EmbeddingViewModel(application: Application) : AndroidViewModel(application) {
    
    private val embeddingModel = EmbeddingModel(application)
    
    private val _uiState = MutableStateFlow(EmbeddingUiState())
    val uiState: StateFlow<EmbeddingUiState> = _uiState.asStateFlow()
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isModelLoading = true, modelLoadError = null)
            
            val result = withContext(Dispatchers.IO) {
                embeddingModel.initialize()
            }
            
            result.fold(
                onSuccess = {
                    _uiState.value = _uiState.value.copy(isModelLoading = false)
                },
                onFailure = { e ->
                    _uiState.value = _uiState.value.copy(
                        isModelLoading = false,
                        modelLoadError = "Failed to load model: ${e.message}"
                    )
                }
            )
        }
    }

    fun updateInputText(text: String) {
        _uiState.value = _uiState.value.copy(inputText = text)
    }
    
    fun generateEmbedding() {
        val text = _uiState.value.inputText.trim()
        if (text.isEmpty()) {
            _uiState.value = _uiState.value.copy(error = "Please enter some text")
            return
        }
        
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isGenerating = true,
                error = null,
                embedding = null
            )
            
            val startTime = System.currentTimeMillis()
            
            val result = withContext(Dispatchers.Default) {
                embeddingModel.generateEmbedding(text, isQuery = true)
            }
            
            val endTime = System.currentTimeMillis()
            
            result.fold(
                onSuccess = { embedding ->
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        embedding = embedding,
                        inferenceTimeMs = endTime - startTime
                    )
                },
                onFailure = { e ->
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        error = "Failed to generate embedding: ${e.message}"
                    )
                }
            )
        }
    }
    
    fun clearEmbedding() {
        _uiState.value = _uiState.value.copy(
            embedding = null,
            error = null,
            inferenceTimeMs = 0
        )
    }
    
    override fun onCleared() {
        super.onCleared()
        embeddingModel.close()
    }
}
