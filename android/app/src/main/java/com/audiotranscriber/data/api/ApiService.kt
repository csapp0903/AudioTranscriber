package com.audiotranscriber.data.api

import com.audiotranscriber.data.model.LoginRequest
import com.audiotranscriber.data.model.LoginResponse
import com.audiotranscriber.data.model.TaskStatusResponse
import com.audiotranscriber.data.model.UploadResponse
import okhttp3.MultipartBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import retrofit2.http.Streaming

/**
 * Retrofit API service interface for AudioTranscriber backend.
 * All suspend functions for coroutine support.
 */
interface ApiService {

    /**
     * User login endpoint.
     * @param loginRequest Contains username and password
     * @return LoginResponse with JWT access token
     */
    @POST("auth/login")
    suspend fun login(
        @Body loginRequest: LoginRequest
    ): Response<LoginResponse>

    /**
     * Upload MP3 file for transcription.
     * Authorization header is automatically added by AuthInterceptor.
     * @param file The MP3 file as MultipartBody.Part
     * @return UploadResponse with task_id
     */
    @Multipart
    @POST("task/upload")
    suspend fun uploadAudio(
        @Part file: MultipartBody.Part
    ): Response<UploadResponse>

    /**
     * Get task status by task ID.
     * @param taskId The task ID returned from upload
     * @return TaskStatusResponse with current status
     */
    @GET("task/status/{task_id}")
    suspend fun getTaskStatus(
        @Path("task_id") taskId: String
    ): Response<TaskStatusResponse>

    /**
     * Download file by task ID and file type.
     * Uses @Streaming for large file downloads to avoid loading entire file into memory.
     * @param taskId The task ID
     * @param fileType File type: pdf, midi, or xml
     * @return ResponseBody for streaming download
     */
    @Streaming
    @GET("task/download/{task_id}/{file_type}")
    suspend fun downloadFile(
        @Path("task_id") taskId: String,
        @Path("file_type") fileType: String
    ): Response<ResponseBody>

    /**
     * Health check endpoint
     */
    @GET("/")
    suspend fun healthCheck(): Response<Map<String, String>>
}
