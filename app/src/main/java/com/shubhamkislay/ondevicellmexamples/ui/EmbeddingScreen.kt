package com.shubhamkislay.ondevicellmexamples.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.rememberAsyncImagePainter
import com.shubhamkislay.ondevicellmexamples.viewmodel.EmbeddingType
import com.shubhamkislay.ondevicellmexamples.viewmodel.EmbeddingViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EmbeddingScreen(
    modifier: Modifier = Modifier,
    viewModel: EmbeddingViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    var selectedTab by remember { mutableIntStateOf(0) }
    
    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.setSelectedImage(uri)
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Nomic Embed v1.5") },
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
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            if (uiState.isModelLoading) {
                ModelLoadingCard()
            } else if (uiState.modelLoadError != null) {
                ErrorCard(message = uiState.modelLoadError!!)
            } else {
                // Tab selector
                TabRow(selectedTabIndex = selectedTab) {
                    Tab(
                        selected = selectedTab == 0,
                        onClick = { selectedTab = 0 },
                        text = { Text("Text") }
                    )
                    Tab(
                        selected = selectedTab == 1,
                        onClick = { selectedTab = 1 },
                        text = { Text("Image") }
                    )
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                
                when (selectedTab) {
                    0 -> TextInputSection(
                        text = uiState.inputText,
                        onTextChange = viewModel::updateInputText,
                        isGenerating = uiState.isGenerating,
                        onGenerate = viewModel::generateTextEmbedding
                    )
                    1 -> ImageInputSection(
                        selectedUri = uiState.selectedImageUri,
                        isGenerating = uiState.isGenerating,
                        onSelectImage = { imagePickerLauncher.launch("image/*") },
                        onGenerate = viewModel::generateImageEmbedding
                    )
                }
                
                // Clear button
                if (uiState.embedding != null) {
                    OutlinedButton(
                        onClick = viewModel::clearEmbedding,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Clear Result")
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
                        embeddingType = uiState.embeddingType,
                        inferenceTimeMs = uiState.inferenceTimeMs
                    )
                }
            }
        }
    }
}

@Composable
private fun TextInputSection(
    text: String,
    onTextChange: (String) -> Unit,
    isGenerating: Boolean,
    onGenerate: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
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
        
        Button(
            onClick = onGenerate,
            enabled = !isGenerating && text.isNotBlank(),
            modifier = Modifier.fillMaxWidth()
        ) {
            if (isGenerating) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    color = MaterialTheme.colorScheme.onPrimary,
                    strokeWidth = 2.dp
                )
                Spacer(modifier = Modifier.width(8.dp))
            }
            Text(if (isGenerating) "Generating..." else "Generate Text Embedding")
        }
    }
}

@Composable
private fun ImageInputSection(
    selectedUri: Uri?,
    isGenerating: Boolean,
    onSelectImage: () -> Unit,
    onGenerate: () -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        // Image preview
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            if (selectedUri != null) {
                Image(
                    painter = rememberAsyncImagePainter(selectedUri),
                    contentDescription = "Selected image",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            } else {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "No image selected",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
        
        // Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedButton(
                onClick = onSelectImage,
                enabled = !isGenerating,
                modifier = Modifier.weight(1f)
            ) {
                Text("Select Image")
            }
            
            Button(
                onClick = onGenerate,
                enabled = !isGenerating && selectedUri != null,
                modifier = Modifier.weight(1f)
            ) {
                if (isGenerating) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        color = MaterialTheme.colorScheme.onPrimary,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(if (isGenerating) "Generating..." else "Embed Image")
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
                text = "Loading models...",
                style = MaterialTheme.typography.bodyLarge
            )
            Text(
                text = "Loading text and vision models",
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
private fun EmbeddingResultCard(
    embedding: FloatArray,
    embeddingType: EmbeddingType?,
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
                Column {
                    Text(
                        text = when (embeddingType) {
                            EmbeddingType.TEXT -> "Text Embedding"
                            EmbeddingType.IMAGE -> "Image Embedding"
                            null -> "Embedding Result"
                        },
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    Text(
                        text = "Dimensions: ${embedding.size}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                }
                Text(
                    text = "${inferenceTimeMs}ms",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer
                )
            }
            
            HorizontalDivider()
            
            // First 10 values preview
            Text(
                text = "First 10 values:",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
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
            
            HorizontalDivider()
            
            // Full vector (scrollable)
            Text(
                text = "Full vector:",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            
            SelectionContainer {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(150.dp)
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
        }
    }
}
