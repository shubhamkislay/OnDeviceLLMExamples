package com.shubhamkislay.ondevicellmexamples.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.shubhamkislay.ondevicellmexamples.model.EmbeddingModel
import com.shubhamkislay.ondevicellmexamples.model.VisionEmbeddingModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class EmbeddingType {
    TEXT, IMAGE
}

data class EmbeddingUiState(
    val isModelLoading: Boolean = true,
    val isGenerating: Boolean = false,
    val modelLoadError: String? = null,
    val inputText: String = "",
    val selectedImageUri: Uri? = null,
    val embedding: FloatArray? = null,
    val embeddingType: EmbeddingType? = null,
    val error: String? = null,
    val inferenceTimeMs: Long = 0
)

class EmbeddingViewModel(application: Application) : AndroidViewModel(application) {
    
    private val textModel = EmbeddingModel(application)
    private val visionModel = VisionEmbeddingModel(application)
    
    private val _uiState = MutableStateFlow(EmbeddingUiState())
    val uiState: StateFlow<EmbeddingUiState> = _uiState.asStateFlow()
    
    init {
        loadModels()
    }
    
    private fun loadModels() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isModelLoading = true, modelLoadError = null)
            
            val textResult = withContext(Dispatchers.IO) {
                textModel.initialize()
            }
            
            val visionResult = withContext(Dispatchers.IO) {
                visionModel.initialize()
            }
            
            val errors = mutableListOf<String>()
            textResult.onFailure { errors.add("Text model: ${it.message}") }
            visionResult.onFailure { errors.add("Vision model: ${it.message}") }
            
            if (errors.isEmpty()) {
                _uiState.value = _uiState.value.copy(isModelLoading = false)
            } else {
                _uiState.value = _uiState.value.copy(
                    isModelLoading = false,
                    modelLoadError = "Failed to load: ${errors.joinToString("; ")}"
                )
            }
        }
    }

    fun updateInputText(text: String) {
        _uiState.value = _uiState.value.copy(inputText = text)
    }
    
    fun setSelectedImage(uri: Uri?) {
        _uiState.value = _uiState.value.copy(selectedImageUri = uri)
    }
    
    fun generateTextEmbedding() {
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
                textModel.generateEmbedding(text, isQuery = true)
            }
            
            val endTime = System.currentTimeMillis()
            
            result.fold(
                onSuccess = { embedding ->
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        embedding = embedding,
                        embeddingType = EmbeddingType.TEXT,
                        inferenceTimeMs = endTime - startTime
                    )
                },
                onFailure = { e ->
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        error = "Failed to generate text embedding: ${e.message}"
                    )
                }
            )
        }
    }
    
    fun generateImageEmbedding() {
        val uri = _uiState.value.selectedImageUri
        if (uri == null) {
            _uiState.value = _uiState.value.copy(error = "Please select an image")
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
                visionModel.generateEmbedding(uri)
            }
            
            val endTime = System.currentTimeMillis()
            
            result.fold(
                onSuccess = { embedding ->
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        embedding = embedding,
                        embeddingType = EmbeddingType.IMAGE,
                        inferenceTimeMs = endTime - startTime
                    )
                },
                onFailure = { e ->
                    Log.e("Embedding",e.message.toString())
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        error = "Failed to generate image embedding: ${e.message}"
                    )
                }
            )
        }
    }
    
    // Keep for backward compatibility
    fun generateEmbedding() = generateTextEmbedding()
    
    fun clearEmbedding() {
        _uiState.value = _uiState.value.copy(
            embedding = null,
            embeddingType = null,
            error = null,
            inferenceTimeMs = 0
        )
    }
    
    override fun onCleared() {
        super.onCleared()
        textModel.close()
        visionModel.close()
    }
}
