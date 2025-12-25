package com.audiotranscriber

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * Application class for AudioTranscriber.
 * Annotated with @HiltAndroidApp to enable Hilt dependency injection.
 */
@HiltAndroidApp
class AudioTranscriberApp : Application()
