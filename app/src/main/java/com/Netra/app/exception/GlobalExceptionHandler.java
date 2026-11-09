package com.Netra.app.exception;

import com.Netra.app.dto.response.ApiResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.stream.Collectors;

/**
 * Centralised exception handler for the NETRA backend.
 *
 * <p>All exceptions thrown within {@code @RestController} methods are caught here
 * and transformed into a uniform {@link ApiResponse} JSON structure. This ensures
 * that clients always receive a consistent error envelope regardless of the
 * specific failure.
 *
 * <p>Handler precedence (most-specific first):
 * <ol>
 *   <li>{@link InvalidImageException}  → 400</li>
 *   <li>{@link FileSizeLimitExceededException} → 413</li>
 *   <li>{@link MaxUploadSizeExceededException} → 413</li>
 *   <li>{@link MissingServletRequestPartException} → 400</li>
 *   <li>{@link MethodArgumentNotValidException} → 422</li>
 *   <li>{@link MethodArgumentTypeMismatchException} → 400</li>
 *   <li>{@link Exception} (catch-all) → 500</li>
 * </ol>
 */
@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    // -------------------------------------------------------------------------
    // Domain exceptions
    // -------------------------------------------------------------------------

    /**
     * Handles validation failures for uploaded images (empty file, bad MIME type).
     */
    @ExceptionHandler(InvalidImageException.class)
    public ResponseEntity<ApiResponse<Void>> handleInvalidImage(InvalidImageException ex) {
        log.warn("InvalidImageException: {}", ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(ApiResponse.error(ex.getMessage()));
    }

    /**
     * Handles programmatic file size limit violations raised within the service layer.
     */
    @ExceptionHandler(FileSizeLimitExceededException.class)
    public ResponseEntity<ApiResponse<Void>> handleFileSizeLimitExceeded(FileSizeLimitExceededException ex) {
        log.warn("FileSizeLimitExceededException: actual={}B, max={}B", ex.getActualSizeBytes(), ex.getMaxSizeBytes());
        return ResponseEntity
                .status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(ApiResponse.error(ex.getMessage()));
    }

    /**
     * Handles failures when communicating with the Python/FastAPI CV engine.
     *
     * <p>Distinguishes two cases:
     * <ul>
     *   <li><b>Engine HTTP error</b> (status code from the engine itself, e.g. 400/422) →
     *       {@code 502 Bad Gateway}: the CV engine received our request but rejected it.</li>
     *   <li><b>Connectivity failure</b> (503 set by {@link com.Netra.app.client.CvEngineClient}) →
     *       {@code 503 Service Unavailable}: the engine is down or unreachable.</li>
     * </ul>
     */
    @ExceptionHandler(com.Netra.app.exception.CvEngineException.class)
    public ResponseEntity<ApiResponse<Void>> handleCvEngineException(
            com.Netra.app.exception.CvEngineException ex) {
        if (ex.getStatusCode() == 503) {
            log.error("CvEngineException [503 — unreachable]: {}", ex.getMessage());
            return ResponseEntity
                    .status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(ApiResponse.error(
                            "The image enhancement engine is currently unavailable. "
                            + "Please try again in a moment."
                    ));
        }
        log.warn("CvEngineException [{}]: {}", ex.getStatusCode(), ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.BAD_GATEWAY)
                .body(ApiResponse.error(
                        "The CV engine could not process this image: " + ex.getMessage()
                ));
    }

    // -------------------------------------------------------------------------
    // Spring multipart exceptions
    // -------------------------------------------------------------------------

    /**
     * Handles Spring's multipart size limit breach (triggered before the
     * controller is reached when {@code spring.servlet.multipart.max-file-size}
     * is exceeded).
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiResponse<Void>> handleMaxUploadSizeExceeded(MaxUploadSizeExceededException ex) {
        log.warn("MaxUploadSizeExceededException: {}", ex.getMessage());
        return ResponseEntity
                .status(HttpStatus.PAYLOAD_TOO_LARGE)
                .body(ApiResponse.error(
                        "The uploaded file exceeds the maximum allowed size. "
                        + "Please reduce the file size and try again."
                ));
    }

    /**
     * Handles missing required multipart parts (e.g. the client forgot to
     * attach the {@code image} part).
     */
    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ApiResponse<Void>> handleMissingPart(MissingServletRequestPartException ex) {
        log.warn("MissingServletRequestPartException: missing part '{}'", ex.getRequestPartName());
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(ApiResponse.error(
                        String.format("Required request part '%s' is missing.", ex.getRequestPartName())
                ));
    }

    // -------------------------------------------------------------------------
    // Bean Validation exceptions
    // -------------------------------------------------------------------------

    /**
     * Handles Bean Validation failures on {@code @Valid}-annotated request bodies or params.
     * Collects all field-level constraint violations into a single readable message.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiResponse<Void>> handleValidationFailed(MethodArgumentNotValidException ex) {
        String violations = ex.getBindingResult().getFieldErrors().stream()
                .map(fe -> String.format("'%s' %s", fe.getField(), fe.getDefaultMessage()))
                .collect(Collectors.joining("; "));
        log.warn("MethodArgumentNotValidException: {}", violations);
        return ResponseEntity
                .status(HttpStatus.UNPROCESSABLE_ENTITY)
                .body(ApiResponse.error("Validation failed: " + violations));
    }

    /**
     * Handles type mismatch errors for path variables and request parameters
     * (e.g. passing a string where a number is expected).
     */
    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ApiResponse<Void>> handleTypeMismatch(MethodArgumentTypeMismatchException ex) {
        String message = String.format(
                "Parameter '%s' has an invalid value '%s'.", ex.getName(), ex.getValue()
        );
        log.warn("MethodArgumentTypeMismatchException: {}", message);
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(ApiResponse.error(message));
    }

    // -------------------------------------------------------------------------
    // Catch-all
    // -------------------------------------------------------------------------

    /**
     * Last-resort handler for any unhandled exception.
     *
     * <p>Returns a generic error message to the client without leaking internal
     * stack trace details. Full details are logged server-side.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiResponse<Void>> handleGenericException(Exception ex) {
        log.error("Unhandled exception in NETRA controller layer", ex);
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ApiResponse.error(
                        "An unexpected error occurred. Please try again later or contact support."
                ));
    }
}
