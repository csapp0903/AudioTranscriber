package com.audiotranscriber.data.network

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages JWT token storage securely using EncryptedSharedPreferences.
 * Provides reactive token state for observing authentication changes.
 */
@Singleton
class TokenManager @Inject constructor(
    private val context: Context
) {
    companion object {
        private const val PREFS_FILE_NAME = "secure_auth_prefs"
        private const val KEY_ACCESS_TOKEN = "access_token"
        private const val KEY_TOKEN_TYPE = "token_type"
    }

    private val _isAuthenticated = MutableStateFlow(false)
    val isAuthenticated: StateFlow<Boolean> = _isAuthenticated.asStateFlow()

    private val encryptedPrefs: SharedPreferences by lazy {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()

        EncryptedSharedPreferences.create(
            context,
            PREFS_FILE_NAME,
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }

    init {
        // Check if token exists on initialization
        _isAuthenticated.value = getAccessToken() != null
    }

    /**
     * Save JWT token securely
     */
    fun saveToken(accessToken: String, tokenType: String = "bearer") {
        encryptedPrefs.edit().apply {
            putString(KEY_ACCESS_TOKEN, accessToken)
            putString(KEY_TOKEN_TYPE, tokenType)
            apply()
        }
        _isAuthenticated.value = true
    }

    /**
     * Get stored access token
     */
    fun getAccessToken(): String? {
        return encryptedPrefs.getString(KEY_ACCESS_TOKEN, null)
    }

    /**
     * Get token type (usually "bearer")
     */
    fun getTokenType(): String {
        return encryptedPrefs.getString(KEY_TOKEN_TYPE, "bearer") ?: "bearer"
    }

    /**
     * Get formatted authorization header value
     */
    fun getAuthorizationHeader(): String? {
        val token = getAccessToken() ?: return null
        val type = getTokenType()
        return "$type $token"
    }

    /**
     * Clear all stored tokens (logout)
     */
    fun clearToken() {
        encryptedPrefs.edit().clear().apply()
        _isAuthenticated.value = false
    }

    /**
     * Check if user has valid token
     */
    fun hasToken(): Boolean = getAccessToken() != null
}
