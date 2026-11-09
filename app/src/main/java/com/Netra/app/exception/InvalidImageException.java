package com.Netra.app.exception;

/**
 * Thrown when an uploaded file fails image validation.
 *
 * <p>Validation failures include, but are not limited to:
 * <ul>
 *   <li>Empty or zero-byte files</li>
 *   <li>Unsupported MIME types</li>
 *   <li>Corrupt or unreadable image data</li>
 * </ul>
 *
 * <p>This exception is handled by {@link GlobalExceptionHandler} and mapped
 * to HTTP {@code 400 Bad Request}.
 */
public class InvalidImageException extends RuntimeException {

    /**
     * Constructs an {@code InvalidImageException} with the given detail message.
     *
     * @param message human-readable description of the validation failure
     */
    public InvalidImageException(String message) {
        super(message);
    }

    /**
     * Constructs an {@code InvalidImageException} with a detail message
     * and an underlying cause.
     *
     * @param message human-readable description of the validation failure
     * @param cause   the original exception that triggered this failure
     */
    public InvalidImageException(String message, Throwable cause) {
        super(message, cause);
    }
}
