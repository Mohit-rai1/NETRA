package com.Netra.app.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Builder;
import lombok.Getter;

import java.time.Instant;

/**
 * Generic API response envelope for all NETRA endpoints.
 *
 * <p>All REST responses are wrapped in this structure to ensure a consistent
 * JSON contract with clients, regardless of success or error state.
 *
 * @param <T> the type of the payload carried in {@code data}
 */
@Getter
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiResponse<T> {

    /** HTTP-level status string, e.g. "SUCCESS" or "ERROR". */
    private final String status;

    /** Human-readable message describing the outcome. */
    private final String message;

    /**
     * Optional payload. {@code null} for error responses; populated with
     * the domain object for success responses.
     */
    private final T data;

    /** Server-side timestamp at the moment the response was constructed. */
    @Builder.Default
    private final Instant timestamp = Instant.now();

    // -------------------------------------------------------------------------
    // Static factory helpers
    // -------------------------------------------------------------------------

    /**
     * Builds a standard success response wrapping the given payload.
     *
     * @param message human-readable success message
     * @param data    the response payload
     * @param <T>     payload type
     * @return a fully populated success {@code ApiResponse}
     */
    public static <T> ApiResponse<T> success(String message, T data) {
        return ApiResponse.<T>builder()
                .status("SUCCESS")
                .message(message)
                .data(data)
                .build();
    }

    /**
     * Builds a standard error response with no data payload.
     *
     * @param message human-readable error description
     * @param <T>     unused payload type (will be {@code null})
     * @return an error {@code ApiResponse}
     */
    public static <T> ApiResponse<T> error(String message) {
        return ApiResponse.<T>builder()
                .status("ERROR")
                .message(message)
                .build();
    }
}
