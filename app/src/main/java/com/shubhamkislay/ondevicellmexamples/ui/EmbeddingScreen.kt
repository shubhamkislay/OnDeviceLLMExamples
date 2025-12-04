package com.shubhamkislay.ondevicellmexamples.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.shubhamkislay.ondevicellmexamples.viewmodel.EmbeddingViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EmbeddingScreen(
    modifier: Modifier = Modifier,
    viewModel: EmbeddingViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Nomic Embed Text v1.5") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        },
        modifier = modifier
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Model loading state
            if (uiState.isModelLoading) {
                ModelLoadingCard()
            } else if (uiState.modelLoadError != null) {
                ErrorCard(message = uiState.modelLoadError!!)
            } else {
                // Input section
                InputSection(
                    text = uiState.inputText,
                    onTextChange = viewModel::updateInputText,
                    isGenerating = uiState.isGenerating
                )

                // Action buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = viewModel::generateEmbedding,
                        enabled = !uiState.isGenerating && uiState.inputText.isNotBlank(),
                        modifier = Modifier.weight(1f)
                    ) {
                        if (uiState.isGenerating) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(20.dp),
                                color = MaterialTheme.colorScheme.onPrimary,
                                strokeWidth = 2.dp
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                        }
                        Text(if (uiState.isGenerating) "Generating..." else "Generate Embedding")
                    }
                    
                    OutlinedButton(
                        onClick = viewModel::clearEmbedding,
                        enabled = uiState.embedding != null
                    ) {
                        Text("Clear")
                    }
                }
                
                // Error display
                uiState.error?.let { error ->
                    ErrorCard(message = error)
                }
                
                // Results section
                uiState.embedding?.let { embedding ->
                    EmbeddingResultCard(
                        embedding = embedding,
                        inferenceTimeMs = uiState.inferenceTimeMs
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelLoadingCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator()
            Text(
                text = "Loading ONNX model...",
                style = MaterialTheme.typography.bodyLarge
            )
            Text(
                text = "This may take a moment on first launch",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun ErrorCard(message: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer
        )
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(16.dp),
            color = MaterialTheme.colorScheme.onErrorContainer
        )
    }
}

@Composable
private fun InputSection(
    text: String,
    onTextChange: (String) -> Unit,
    isGenerating: Boolean
) {
    OutlinedTextField(
        value = text,
        onValueChange = onTextChange,
        label = { Text("Enter text to embed") },
        placeholder = { Text("Type your text here...") },
        modifier = Modifier
            .fillMaxWidth()
            .height(120.dp),
        enabled = !isGenerating,
        maxLines = 5
    )
}


@Composable
private fun EmbeddingResultCard(
    embedding: FloatArray,
    inferenceTimeMs: Long
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Header with stats
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Embedding Result",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer
                )
                Text(
                    text = "${inferenceTimeMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer
                )
            }
            
            // Dimension info
            Text(
                text = "Dimensions: ${embedding.size}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            
            HorizontalDivider()
            
            // Embedding values (scrollable)
            Text(
                text = "Vector values:",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            
            SelectionContainer {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .verticalScroll(rememberScrollState())
                ) {
                    Text(
                        text = embedding.joinToString(separator = ", ") { 
                            String.format("%.6f", it) 
                        },
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp,
                            lineHeight = 14.sp
                        ),
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
            }
            
            // First few values preview
            HorizontalDivider()
            
            Text(
                text = "First 10 values:",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            
            Column(
                verticalArrangement = Arrangement.spacedBy(2.dp)
            ) {
                embedding.take(10).forEachIndexed { index, value ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "[$index]",
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace
                            ),
                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.7f)
                        )
                        Text(
                            text = String.format("%.8f", value),
                            style = MaterialTheme.typography.bodySmall.copy(
                                fontFamily = FontFamily.Monospace
                            ),
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                    }
                }
            }
        }
    }
}
