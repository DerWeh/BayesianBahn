package io.github.derweh.bayesianbahn.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

private val LightColors = lightColorScheme(
    primary = Color(0xFF1A5276),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD0E4F2),
    onPrimaryContainer = Color(0xFF0B2A3E),
    secondary = Color(0xFF5B6B75),
    surfaceVariant = Color(0xFFE4EAEE),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF8FC3E4),
    onPrimary = Color(0xFF0B2A3E),
    primaryContainer = Color(0xFF15405D),
    onPrimaryContainer = Color(0xFFD0E4F2),
    secondary = Color(0xFFA9B8C2),
)

@Composable
fun BayesianBahnTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val colorScheme = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val context = LocalContext.current
        if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
    } else {
        if (dark) DarkColors else LightColors
    }
    MaterialTheme(colorScheme = colorScheme, content = content)
}
