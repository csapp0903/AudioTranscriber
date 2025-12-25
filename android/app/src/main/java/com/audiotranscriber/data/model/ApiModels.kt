package com.audiotranscriber.data.model

import com.google.gson.annotations.SerializedName

/**
 * Login request body
 */
data class LoginRequest(
    @SerializedName("username") val username: String,
    @SerializedName("password") val password: String
)

/**
 * Login response containing JWT token
 */
data class LoginResponse(
    @SerializedName("access_token") val accessToken: String,
    @SerializedName("token_type") val tokenType: String = "bearer"
)

/**
 * Upload response containing task ID
 */
data class UploadResponse(
    @SerializedName("task_id") val taskId: String,
    @SerializedName("filename") val filename: String? = null,
    @SerializedName("message") val message: String? = null
)

/**
 * Task status enum
 */
enum class TaskStatus {
    @SerializedName("PENDING") PENDING,
    @SerializedName("PROCESSING") PROCESSING,
    @SerializedName("SUCCESS") SUCCESS,
    @SerializedName("FAILURE") FAILURE
}

/**
 * Task status response
 */
data class TaskStatusResponse(
    @SerializedName("task_id") val taskId: String,
    @SerializedName("status") val status: TaskStatus,
    @SerializedName("result") val result: TaskResult? = null,
    @SerializedName("error") val error: String? = null,
    @SerializedName("meta") val meta: Map<String, Any>? = null
)

/**
 * Task result containing file paths
 */
data class TaskResult(
    @SerializedName("files") val files: Map<String, String>? = null,
    @SerializedName("available_files") val availableFiles: List<String>? = null,
    @SerializedName("download_urls") val downloadUrls: Map<String, String>? = null
)

/**
 * Download file type enum
 */
enum class FileType(val value: String) {
    PDF("pdf"),
    MIDI("midi"),
    XML("xml")
}

/**
 * Sealed class for polling state
 */
sealed class PollingState {
    data object Idle : PollingState()
    data object Pending : PollingState()
    data object Processing : PollingState()
    data class Success(val result: TaskResult?) : PollingState()
    data class Error(val message: String) : PollingState()
    data class Timeout(val taskId: String) : PollingState()
}

/**
 * Sealed class for download result
 */
sealed class DownloadResult {
    data class Success(val filePath: String) : DownloadResult()
    data class Error(val message: String) : DownloadResult()
    data class Progress(val percentage: Int) : DownloadResult()
}
