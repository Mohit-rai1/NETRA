package com.Netra.app.exception;

/**
 * Thrown when an uploaded file exceeds the application-defined size limit.
 *
 * <p>Note: Spring's {@link org.springframework.web.multipart.MaxUploadSizeExceededException}
 * is caught by {@link GlobalExceptionHandler} and mapped to HTTP {@code 413}.
 * This exception is for programmatic size checks within the service layer
 * that sit below the Spring multipart threshold.
 */
public class FileSizeLimitExceededException extends RuntimeException {

    /** The actual size of the file that was rejected, in bytes. */
    private final long actualSizeBytes;

    /** The maximum permitted size, in bytes. */
    private final long maxSizeBytes;

    /**
     * Constructs a {@code FileSizeLimitExceededException}.
     *
     * @param actualSizeBytes the size of the file that was rejected
     * @param maxSizeBytes    the configured maximum allowed size
     */
    public FileSizeLimitExceededException(long actualSizeBytes, long maxSizeBytes) {
        super(String.format(
                "File size %d bytes exceeds the maximum allowed size of %d bytes.",
                actualSizeBytes, maxSizeBytes
        ));
        this.actualSizeBytes = actualSizeBytes;
        this.maxSizeBytes = maxSizeBytes;
    }

    /** Returns the actual file size in bytes that caused this exception. */
    public long getActualSizeBytes() {
        return actualSizeBytes;
    }

    /** Returns the maximum permitted file size in bytes. */
    public long getMaxSizeBytes() {
        return maxSizeBytes;
    }
}
