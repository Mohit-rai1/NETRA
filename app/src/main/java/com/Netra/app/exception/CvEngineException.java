package com.Netra.app.exception;

/**
 * Thrown when communication with the Python/FastAPI CV engine fails.
 *
 * <p>Covers two failure modes:
 * <ul>
 *   <li><b>HTTP errors</b> — the engine responded with a 4xx/5xx status
 *       (e.g. invalid image rejected by the Python layer).</li>
 *   <li><b>Connection errors</b> — the engine is unreachable (503-class).</li>
 * </ul>
 *
 * <p>Handled by {@link GlobalExceptionHandler} and mapped to an appropriate
 * HTTP status code for the upstream caller.
 */
public class CvEngineException extends RuntimeException {

    /** HTTP status code returned by (or inferred for) the CV engine. */
    private final int statusCode;

    /**
     * Constructs a {@code CvEngineException}.
     *
     * @param statusCode HTTP status code from the CV engine response
     * @param message    human-readable error description
     */
    public CvEngineException(int statusCode, String message) {
        super(message);
        this.statusCode = statusCode;
    }

    /**
     * Constructs a {@code CvEngineException} with an underlying cause.
     *
     * @param statusCode HTTP status code from the CV engine response
     * @param message    human-readable error description
     * @param cause      the original exception (e.g. {@code ResourceAccessException})
     */
    public CvEngineException(int statusCode, String message, Throwable cause) {
        super(message, cause);
        this.statusCode = statusCode;
    }

    /**
     * Returns the HTTP status code associated with this failure.
     *
     * <p>503 is used when the engine is unreachable (connection refused / timeout).
     * Values like 400, 413, 422 reflect errors returned by the engine itself.
     */
    public int getStatusCode() {
        return statusCode;
    }
}
