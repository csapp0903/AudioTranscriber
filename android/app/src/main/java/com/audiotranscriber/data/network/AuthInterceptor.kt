package com.audiotranscriber.data.network

import okhttp3.Interceptor
import okhttp3.Response
import javax.inject.Inject
import javax.inject.Singleton

/**
 * OkHttp Interceptor that automatically adds JWT Authorization header to requests.
 * Skips auth header for login endpoint.
 */
@Singleton
class AuthInterceptor @Inject constructor(
    private val tokenManager: TokenManager
) : Interceptor {

    companion object {
        private const val HEADER_AUTHORIZATION = "Authorization"
        private val NO_AUTH_ENDPOINTS = listOf(
            "/auth/login",
            "/auth/register"
        )
    }

    override fun intercept(chain: Interceptor.Chain): Response {
        val originalRequest = chain.request()

        // Check if this endpoint requires authentication
        val shouldSkipAuth = NO_AUTH_ENDPOINTS.any { endpoint ->
            originalRequest.url.encodedPath.endsWith(endpoint)
        }

        // If no auth required or no token available, proceed without auth header
        if (shouldSkipAuth) {
            return chain.proceed(originalRequest)
        }

        val authHeader = tokenManager.getAuthorizationHeader()

        // If no token available, proceed without auth header
        // The server will return 401 if authentication is required
        if (authHeader == null) {
            return chain.proceed(originalRequest)
        }

        // Add Authorization header
        val authenticatedRequest = originalRequest.newBuilder()
            .header(HEADER_AUTHORIZATION, authHeader)
            .build()

        return chain.proceed(authenticatedRequest)
    }
}
