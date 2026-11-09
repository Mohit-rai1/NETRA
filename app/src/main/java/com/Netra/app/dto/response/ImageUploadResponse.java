package com.Netra.app.dto.response;

import lombok.Builder;
import lombok.Getter;

import java.time.Instant;

/**
 * Response payload returned by the image enhancement upload endpoint.
 *
 * <p>This DTO carries the acknowledgment metadata for a successfully received
 * image file. The enhancement process itself is asynchronous — the CV engine
 * orchestration will be tracked via {@code imageId} in subsequent calls.
 */
@Getter
@Builder
public class ImageUploadResponse {

    /**
     * Unique identifier assigned to this upload session.
     * Used to poll for the enhancement result once the CV engine completes.
     */
    private final String imageId;

    /** The original filename as provided by the client. */
    private final String originalFilename;

    /** MIME content type of the uploaded file, e.g. {@code image/jpeg}. */
    private final String contentType;

    /** Size of the uploaded file in bytes. */
    private final long fileSizeBytes;

    /** Server-side timestamp at which the file was successfully received. */
    @Builder.Default
    private final Instant uploadedAt = Instant.now();

    /**
     * Processing status of this upload.
     * Initially {@code "RECEIVED"}; transitions to {@code "PROCESSING"},
     * {@code "COMPLETED"}, or {@code "FAILED"} as the CV pipeline progresses.
     */
    @Builder.Default
    private final String status = "RECEIVED";
}
