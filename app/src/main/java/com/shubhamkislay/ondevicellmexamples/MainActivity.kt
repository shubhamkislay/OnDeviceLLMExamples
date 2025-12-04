package com.shubhamkislay.ondevicellmexamples

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import com.shubhamkislay.ondevicellmexamples.ui.EmbeddingScreen
import com.shubhamkislay.ondevicellmexamples.ui.theme.OnDeviceLLMExamplesTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            OnDeviceLLMExamplesTheme {
                EmbeddingScreen(modifier = Modifier.fillMaxSize())
            }
        }
    }
}
