package com.Netra.app.dto.response;

/**
 * Result produced by the CV engine call and returned to the controller.
 *
 * <p>Carries the raw image bytes of the enhanced image alongside metadata
 * extracted from the Python engine's custom response headers.
 *
 * @param imageBytes        Raw bytes of the enhanced image (JPEG or PNG).
 * @param contentType       MIME type of the returned image, e.g. {@code image/jpeg}.
 * @param originalFilename  The original filename as uploaded by the client.
 * @param stepsApplied      Comma-separated pipeline steps executed by the engine,
 *                          e.g. {@code "decode, gray_world(...), clahe_lab(...)"}.
 * @param processingTimeMs  End-to-end processing time in the Python engine,
 *                          as a string, e.g. {@code "14.3"}.
 */
public record EnhancedImageResult(
        byte[]  imageBytes,
        String  contentType,
        String  originalFilename,
        String  stepsApplied,
        String  processingTimeMs
) {}
