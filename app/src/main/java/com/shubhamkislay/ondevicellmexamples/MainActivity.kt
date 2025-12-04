package com.shubhamkislay.ondevicellmexamples

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.shubhamkislay.ondevicellmexamples.ui.EmbeddingScreen
import com.shubhamkislay.ondevicellmexamples.ui.theme.OnDeviceLLMExamplesTheme
import com.shubhamkislay.ondevicellmexamples.viewmodel.EmbeddingViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OnDeviceLLMExamplesTheme {
                val viewModel: EmbeddingViewModel = viewModel()
                val uiState by viewModel.uiState.collectAsState()
                
                EmbeddingScreen(
                    uiState = uiState,
                    onTextChange = viewModel::updateInputText,
                    onGenerateClick = viewModel::generateEmbedding,
                    onClearClick = viewModel::clearEmbedding,
                    modifier = Modifier.fillMaxSize()
                )
            }
        }
    }
}
