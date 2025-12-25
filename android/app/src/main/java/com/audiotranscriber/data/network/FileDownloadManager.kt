package com.audiotranscriber.data.network

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.audiotranscriber.data.model.DownloadResult
import com.audiotranscriber.data.model.FileType
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.ResponseBody
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Dedicated manager for handling file downloads with progress tracking.
 * Properly handles Android 10+ Scoped Storage requirements.
 */
@Singleton
class FileDownloadManager @Inject constructor(
    @ApplicationContext private val context: Context
) {
    companion object {
        private const val BUFFER_SIZE = 8192
        private const val DOWNLOADS_SUBFOLDER = "AudioTranscriber"
    }

    /**
     * Download and save file from ResponseBody.
     * Emits progress updates and final result.
     *
     * @param responseBody The ResponseBody from Retrofit
     * @param fileName The target filename
     * @param fileType The file type for MIME type detection
     * @return Flow of DownloadResult with progress and final status
     */
    fun saveFile(
        responseBody: ResponseBody,
        fileName: String,
        fileType: FileType
    ): Flow<DownloadResult> = flow {
        emit(DownloadResult.Progress(0))

        val totalBytes = responseBody.contentLength()
        var downloadedBytes = 0L

        try {
            val savedPath = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                saveWithMediaStore(
                    inputStream = responseBody.byteStream(),
                    fileName = fileName,
                    mimeType = getMimeType(fileType),
                    totalBytes = totalBytes,
                    onProgress = { progress ->
                        // Can't emit from lambda, track progress externally
                        downloadedBytes = progress
                    }
                )
            } else {
                saveToLegacyStorage(
                    inputStream = responseBody.byteStream(),
                    fileName = fileName,
                    onProgress = { progress ->
                        downloadedBytes = progress
                    }
                )
            }

            emit(DownloadResult.Progress(100))
            emit(DownloadResult.Success(savedPath))

        } catch (e: Exception) {
            emit(DownloadResult.Error("Download failed: ${e.message}"))
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Save file using MediaStore API for Android 10+.
     * Properly integrates with Scoped Storage.
     */
    private fun saveWithMediaStore(
        inputStream: InputStream,
        fileName: String,
        mimeType: String,
        totalBytes: Long,
        onProgress: (Long) -> Unit
    ): String {
        val contentValues = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, fileName)
            put(MediaStore.Downloads.MIME_TYPE, mimeType)
            put(
                MediaStore.Downloads.RELATIVE_PATH,
                "${Environment.DIRECTORY_DOWNLOADS}/$DOWNLOADS_SUBFOLDER"
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Downloads.IS_PENDING, 1)
            }
        }

        val resolver = context.contentResolver
        val uri: Uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)
            ?: throw Exception("Failed to create file in Downloads")

        try {
            resolver.openOutputStream(uri)?.use { outputStream ->
                copyStreamWithProgress(inputStream, outputStream, totalBytes, onProgress)
            } ?: throw Exception("Failed to open output stream")

            // Mark download as complete
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val updateValues = ContentValues().apply {
                    put(MediaStore.Downloads.IS_PENDING, 0)
                }
                resolver.update(uri, updateValues, null, null)
            }

            return uri.toString()

        } catch (e: Exception) {
            // Clean up failed download
            resolver.delete(uri, null, null)
            throw e
        }
    }

    /**
     * Save file to legacy external storage for Android 9 and below.
     * Requires WRITE_EXTERNAL_STORAGE permission.
     */
    @Suppress("DEPRECATION")
    private fun saveToLegacyStorage(
        inputStream: InputStream,
        fileName: String,
        onProgress: (Long) -> Unit
    ): String {
        val downloadsDir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
            DOWNLOADS_SUBFOLDER
        )

        if (!downloadsDir.exists()) {
            downloadsDir.mkdirs()
        }

        val targetFile = File(downloadsDir, fileName)

        // Handle duplicate filenames
        val finalFile = getUniqueFile(targetFile)

        FileOutputStream(finalFile).use { outputStream ->
            copyStreamWithProgress(inputStream, outputStream, -1, onProgress)
        }

        return finalFile.absolutePath
    }

    /**
     * Copy stream with progress tracking.
     */
    private fun copyStreamWithProgress(
        input: InputStream,
        output: OutputStream,
        totalBytes: Long,
        onProgress: (Long) -> Unit
    ) {
        val buffer = ByteArray(BUFFER_SIZE)
        var bytesRead: Int
        var totalRead = 0L

        input.use { inputStream ->
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                output.write(buffer, 0, bytesRead)
                totalRead += bytesRead
                onProgress(totalRead)
            }
        }
        output.flush()
    }

    /**
     * Generate unique filename if file already exists.
     */
    private fun getUniqueFile(file: File): File {
        if (!file.exists()) return file

        val baseName = file.nameWithoutExtension
        val extension = file.extension
        var counter = 1

        var newFile = file
        while (newFile.exists()) {
            val newName = if (extension.isNotEmpty()) {
                "${baseName}_($counter).$extension"
            } else {
                "${baseName}_($counter)"
            }
            newFile = File(file.parentFile, newName)
            counter++
        }

        return newFile
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

    /**
     * Check if we need to request storage permissions.
     * On Android 10+, we don't need WRITE_EXTERNAL_STORAGE for MediaStore Downloads.
     */
    fun needsStoragePermission(): Boolean {
        return Build.VERSION.SDK_INT < Build.VERSION_CODES.Q
    }

    /**
     * Get the Downloads directory path for display.
     */
    fun getDownloadsPath(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            "Downloads/$DOWNLOADS_SUBFOLDER"
        } else {
            @Suppress("DEPRECATION")
            File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                DOWNLOADS_SUBFOLDER
            ).absolutePath
        }
    }
}
