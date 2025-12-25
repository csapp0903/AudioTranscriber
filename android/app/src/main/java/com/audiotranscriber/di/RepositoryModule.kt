package com.audiotranscriber.di

import android.content.Context
import com.audiotranscriber.data.api.ApiService
import com.audiotranscriber.data.network.FileDownloadManager
import com.audiotranscriber.data.network.TokenManager
import com.audiotranscriber.data.repository.TranscriptionRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

/**
 * Hilt module for providing repository dependencies.
 */
@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {

    @Provides
    @Singleton
    fun provideTranscriptionRepository(
        apiService: ApiService,
        tokenManager: TokenManager,
        @ApplicationContext context: Context
    ): TranscriptionRepository {
        return TranscriptionRepository(apiService, tokenManager, context)
    }

    @Provides
    @Singleton
    fun provideFileDownloadManager(
        @ApplicationContext context: Context
    ): FileDownloadManager {
        return FileDownloadManager(context)
    }
}
