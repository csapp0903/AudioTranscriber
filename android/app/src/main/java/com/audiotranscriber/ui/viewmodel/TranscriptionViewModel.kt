package com.audiotranscriber.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.audiotranscriber.data.model.DownloadResult
import com.audiotranscriber.data.model.FileType
import com.audiotranscriber.data.model.PollingState
import com.audiotranscriber.data.model.TaskResult
import com.audiotranscriber.data.repository.TranscriptionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import javax.inject.Inject

/**
 * UI State for transcription screen
 */
data class TranscriptionUiState(
    val isLoading: Boolean = false,
    val isUploading: Boolean = false,
    val uploadProgress: Int = 0,
    val pollingState: PollingState = PollingState.Idle,
    val taskId: String? = null,
    val taskResult: TaskResult? = null,
    val downloadProgress: Int = 0,
    val downloadedFilePath: String? = null,
    val errorMessage: String? = null
)

/**
 * One-time UI events
 */
sealed class TranscriptionEvent {
    data class ShowToast(val message: String) : TranscriptionEvent()
    data class NavigateToResult(val taskId: String) : TranscriptionEvent()
    data class OpenFile(val filePath: String) : TranscriptionEvent()
    data object NavigateToLogin : TranscriptionEvent()
}

/**
 * ViewModel demonstrating how to use TranscriptionRepository in UI layer.
 * Uses Jetpack Compose compatible state management.
 */
@HiltViewModel
class TranscriptionViewModel @Inject constructor(
    private val repository: TranscriptionRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(TranscriptionUiState())
    val uiState: StateFlow<TranscriptionUiState> = _uiState.asStateFlow()

    private val _events = MutableSharedFlow<TranscriptionEvent>()
    val events: SharedFlow<TranscriptionEvent> = _events.asSharedFlow()

    private var pollingJob: Job? = null

    // region Authentication

    fun login(username: String, password: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)

            repository.login(username, password)
                .onSuccess {
                    _uiState.value = _uiState.value.copy(isLoading = false)
                    _events.emit(TranscriptionEvent.ShowToast("Login successful"))
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        errorMessage = error.message
                    )
                }
        }
    }

    fun logout() {
        repository.logout()
        _events.tryEmit(TranscriptionEvent.NavigateToLogin)
    }

    // endregion

    // region Upload

    fun uploadAudioFile(file: File) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isUploading = true,
                errorMessage = null,
                pollingState = PollingState.Idle
            )

            repository.uploadAudio(file)
                .onSuccess { response ->
                    _uiState.value = _uiState.value.copy(
                        isUploading = false,
                        taskId = response.taskId
                    )
                    // Automatically start polling after successful upload
                    startPolling(response.taskId)
                }
                .onFailure { error ->
                    _uiState.value = _uiState.value.copy(
                        isUploading = false,
                        errorMessage = error.message
                    )
                }
        }
    }

    // endregion

    // region Polling

    fun startPolling(taskId: String) {
        // Cancel any existing polling job
        pollingJob?.cancel()

        pollingJob = viewModelScope.launch {
            repository.startPolling(taskId).collect { state ->
                _uiState.value = _uiState.value.copy(pollingState = state)

                when (state) {
                    is PollingState.Success -> {
                        _uiState.value = _uiState.value.copy(taskResult = state.result)
                        _events.emit(TranscriptionEvent.NavigateToResult(taskId))
                    }
                    is PollingState.Error -> {
                        _uiState.value = _uiState.value.copy(errorMessage = state.message)
                    }
                    is PollingState.Timeout -> {
                        _uiState.value = _uiState.value.copy(
                            errorMessage = "Transcription timed out. Please try again."
                        )
                    }
                    else -> {
                        // Pending or Processing - UI will show appropriate loading state
                    }
                }
            }
        }
    }

    fun stopPolling() {
        pollingJob?.cancel()
        pollingJob = null
        _uiState.value = _uiState.value.copy(pollingState = PollingState.Idle)
    }

    // endregion

    // region Download

    fun downloadFile(taskId: String, fileType: FileType) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                downloadProgress = 0,
                errorMessage = null
            )

            repository.downloadFile(taskId, fileType).collect { result ->
                when (result) {
                    is DownloadResult.Progress -> {
                        _uiState.value = _uiState.value.copy(
                            downloadProgress = result.percentage
                        )
                    }
                    is DownloadResult.Success -> {
                        _uiState.value = _uiState.value.copy(
                            downloadedFilePath = result.filePath,
                            downloadProgress = 100
                        )
                        _events.emit(TranscriptionEvent.OpenFile(result.filePath))
                        _events.emit(TranscriptionEvent.ShowToast("File saved to Downloads"))
                    }
                    is DownloadResult.Error -> {
                        _uiState.value = _uiState.value.copy(
                            errorMessage = result.message,
                            downloadProgress = 0
                        )
                    }
                }
            }
        }
    }

    // endregion

    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }

    override fun onCleared() {
        super.onCleared()
        pollingJob?.cancel()
    }
}
