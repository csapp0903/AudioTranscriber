package com.audiotranscriber.data.repository

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.audiotranscriber.data.api.ApiService
import com.audiotranscriber.data.model.DownloadResult
import com.audiotranscriber.data.model.FileType
import com.audiotranscriber.data.model.LoginRequest
import com.audiotranscriber.data.model.LoginResponse
import com.audiotranscriber.data.model.PollingState
import com.audiotranscriber.data.model.TaskStatus
import com.audiotranscriber.data.model.TaskStatusResponse
import com.audiotranscriber.data.model.UploadResponse
import com.audiotranscriber.data.network.TokenManager
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Repository for handling audio transcription operations.
 * Provides methods for login, upload, status polling, and file download.
 */
@Singleton
class TranscriptionRepository @Inject constructor(
    private val apiService: ApiService,
    private val tokenManager: TokenManager,
    @ApplicationContext private val context: Context
) {
    companion object {
        // Polling configuration
        private const val POLLING_INTERVAL_MS = 3000L // 3 seconds
        private const val POLLING_TIMEOUT_MS = 5 * 60 * 1000L // 5 minutes

        // Download buffer size
        private const val BUFFER_SIZE = 8192
    }

    // region Authentication

    /**
     * Login with username and password.
     * Stores JWT token on success.
     * @return Result with LoginResponse on success, or error message on failure
     */
    suspend fun login(username: String, password: String): Result<LoginResponse> {
        return try {
            val response = apiService.login(LoginRequest(username, password))

            if (response.isSuccessful) {
                val loginResponse = response.body()
                    ?: return Result.failure(Exception("Empty response body"))

                // Store token securely
                tokenManager.saveToken(
                    accessToken = loginResponse.accessToken,
                    tokenType = loginResponse.tokenType
                )

                Result.success(loginResponse)
            } else {
                val errorMessage = when (response.code()) {
                    401 -> "Invalid username or password"
                    403 -> "Account locked or disabled"
                    else -> "Login failed: ${response.message()}"
                }
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Result.failure(Exception("Network error: ${e.message}"))
        }
    }

    /**
     * Logout - clears stored token
     */
    fun logout() {
        tokenManager.clearToken()
    }

    /**
     * Check if user is authenticated
     */
    fun isAuthenticated(): Boolean = tokenManager.hasToken()

    // endregion

    // region Upload

    /**
     * Upload an MP3 file for transcription.
     * @param audioFile The MP3 file to upload
     * @return Result with UploadResponse containing task_id
     */
    suspend fun uploadAudio(audioFile: File): Result<UploadResponse> {
        return try {
            if (!audioFile.exists()) {
                return Result.failure(Exception("File not found: ${audioFile.path}"))
            }

            if (!audioFile.name.lowercase().endsWith(".mp3")) {
                return Result.failure(Exception("Only MP3 files are supported"))
            }

            val requestBody = audioFile.asRequestBody("audio/mpeg".toMediaTypeOrNull())
            val multipartBody = MultipartBody.Part.createFormData(
                "file",
                audioFile.name,
                requestBody
            )

            val response = apiService.uploadAudio(multipartBody)

            if (response.isSuccessful) {
                val uploadResponse = response.body()
                    ?: return Result.failure(Exception("Empty response body"))
                Result.success(uploadResponse)
            } else {
                val errorMessage = when (response.code()) {
                    400 -> "Invalid file format"
                    401 -> "Unauthorized - please login again"
                    413 -> "File too large"
                    else -> "Upload failed: ${response.message()}"
                }
                Result.failure(Exception(errorMessage))
            }
        } catch (e: Exception) {
            Result.failure(Exception("Upload error: ${e.message}"))
        }
    }

    // endregion

    // region Status Polling

    /**
     * Start polling for task status.
     * Emits PollingState updates every 3 seconds until completion or timeout.
     *
     * @param taskId The task ID to poll
     * @return Flow of PollingState updates
     *
     * Usage example:
     * ```
     * repository.startPolling(taskId)
     *     .collect { state ->
     *         when (state) {
     *             is PollingState.Success -> handleSuccess(state.result)
     *             is PollingState.Error -> showError(state.message)
     *             is PollingState.Timeout -> showTimeout()
     *             else -> updateUI(state)
     *         }
     *     }
     * ```
     */
    fun startPolling(taskId: String): Flow<PollingState> = flow {
        val startTime = System.currentTimeMillis()

        emit(PollingState.Pending)

        while (true) {
            // Check for timeout
            val elapsedTime = System.currentTimeMillis() - startTime
            if (elapsedTime >= POLLING_TIMEOUT_MS) {
                emit(PollingState.Timeout(taskId))
                return@flow
            }

            try {
                val response = apiService.getTaskStatus(taskId)

                if (response.isSuccessful) {
                    val statusResponse = response.body()
                        ?: throw Exception("Empty status response")

                    when (statusResponse.status) {
                        TaskStatus.PENDING -> {
                            emit(PollingState.Pending)
                        }
                        TaskStatus.PROCESSING -> {
                            emit(PollingState.Processing)
                        }
                        TaskStatus.SUCCESS -> {
                            emit(PollingState.Success(statusResponse.result))
                            return@flow // Stop polling on success
                        }
                        TaskStatus.FAILURE -> {
                            val errorMessage = statusResponse.error
                                ?: "Transcription failed"
                            emit(PollingState.Error(errorMessage))
                            return@flow // Stop polling on failure
                        }
                    }
                } else {
                    when (response.code()) {
                        404 -> {
                            emit(PollingState.Error("Task not found"))
                            return@flow
                        }
                        401 -> {
                            emit(PollingState.Error("Session expired - please login again"))
                            return@flow
                        }
                        else -> {
                            // Transient error - continue polling
                            emit(PollingState.Error("Status check failed: ${response.message()}"))
                        }
                    }
                }
            } catch (e: CancellationException) {
                // Re-throw cancellation to properly cancel the flow
                throw e
            } catch (e: Exception) {
                // Network error - continue polling, may be transient
                emit(PollingState.Error("Network error: ${e.message}"))
            }

            // Wait before next poll
            delay(POLLING_INTERVAL_MS)
        }
    }

    /**
     * Get current task status once (non-polling).
     * @param taskId The task ID
     * @return Result with TaskStatusResponse
     */
    suspend fun getTaskStatus(taskId: String): Result<TaskStatusResponse> {
        return try {
            val response = apiService.getTaskStatus(taskId)

            if (response.isSuccessful) {
                val statusResponse = response.body()
                    ?: return Result.failure(Exception("Empty response"))
                Result.success(statusResponse)
            } else {
                Result.failure(Exception("Failed to get status: ${response.message()}"))
            }
        } catch (e: Exception) {
            Result.failure(Exception("Network error: ${e.message}"))
        }
    }

    // endregion

    // region File Download

    /**
     * Download file to device Downloads folder.
     * Handles Android 10+ Scoped Storage requirements.
     *
     * @param taskId The task ID
     * @param fileType The file type to download (pdf, midi, xml)
     * @return Flow of DownloadResult with progress updates
     */
    fun downloadFile(taskId: String, fileType: FileType): Flow<DownloadResult> = flow {
        try {
            emit(DownloadResult.Progress(0))

            val response = apiService.downloadFile(taskId, fileType.value)

            if (!response.isSuccessful) {
                emit(DownloadResult.Error("Download failed: ${response.message()}"))
                return@flow
            }

            val responseBody = response.body()
                ?: throw Exception("Empty response body")

            // Generate filename
            val fileName = "transcription_${taskId.take(8)}.${fileType.value}"

            // Save file using appropriate method for Android version
            val savedPath = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveFileWithMediaStore(responseBody.byteStream(), fileName, fileType)
            } else {
                saveFileToLegacyStorage(responseBody.byteStream(), fileName)
            }

            emit(DownloadResult.Progress(100))
            emit(DownloadResult.Success(savedPath))

        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            emit(DownloadResult.Error("Download error: ${e.message}"))
        }
    }

    /**
     * Save file using MediaStore API (Android 10+).
     * Properly handles Scoped Storage requirements.
     */
    private fun saveFileWithMediaStore(
        inputStream: InputStream,
        fileName: String,
        fileType: FileType
    ): String {
        val contentValues = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, fileName)
            put(MediaStore.Downloads.MIME_TYPE, getMimeType(fileType))
            put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS)
            // Mark as pending during write
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Downloads.IS_PENDING, 1)
            }
        }

        val resolver = context.contentResolver
        val uri: Uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)
            ?: throw Exception("Failed to create MediaStore entry")

        try {
            resolver.openOutputStream(uri)?.use { outputStream ->
                copyStream(inputStream, outputStream)
            } ?: throw Exception("Failed to open output stream")

            // Mark as complete
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                contentValues.clear()
                contentValues.put(MediaStore.Downloads.IS_PENDING, 0)
                resolver.update(uri, contentValues, null, null)
            }

            return uri.toString()
        } catch (e: Exception) {
            // Clean up on failure
            resolver.delete(uri, null, null)
            throw e
        }
    }

    /**
     * Save file to legacy external storage (Android 9 and below).
     * Requires WRITE_EXTERNAL_STORAGE permission.
     */
    @Suppress("DEPRECATION")
    private fun saveFileToLegacyStorage(
        inputStream: InputStream,
        fileName: String
    ): String {
        val downloadsDir = Environment.getExternalStoragePublicDirectory(
            Environment.DIRECTORY_DOWNLOADS
        )

        if (!downloadsDir.exists()) {
            downloadsDir.mkdirs()
        }

        val file = File(downloadsDir, fileName)

        FileOutputStream(file).use { outputStream ->
            copyStream(inputStream, outputStream)
        }

        return file.absolutePath
    }

    /**
     * Copy input stream to output stream with buffering.
     */
    private fun copyStream(input: InputStream, output: OutputStream) {
        val buffer = ByteArray(BUFFER_SIZE)
        var bytesRead: Int

        input.use { inputStream ->
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                output.write(buffer, 0, bytesRead)
            }
        }
        output.flush()
    }

    /**
     * Get MIME type for file type.
     */
    private fun getMimeType(fileType: FileType): String {
        return when (fileType) {
            FileType.PDF -> "application/pdf"
            FileType.MIDI -> "audio/midi"
            FileType.XML -> "application/xml"
        }
    }

    // endregion
}
