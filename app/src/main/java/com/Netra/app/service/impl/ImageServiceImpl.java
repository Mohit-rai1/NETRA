package com.Netra.app.service.impl;

import com.Netra.app.client.CvEngineClient;
import com.Netra.app.dto.response.EnhancedImageResult;
import com.Netra.app.exception.InvalidImageException;
import com.Netra.app.service.ImageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.Set;

/**
 * Default implementation of {@link ImageService}.
 *
 * <p>Performs client-side validation (empty-file guard + MIME type check), then
 * delegates to {@link CvEngineClient} to run the physics-guided enhancement
 * pipeline on the Python/FastAPI side.  The enhanced image bytes and pipeline
 * metadata are returned to the caller without further transformation.
 *
 * <p>This class is intentionally thin — business logic lives in the Python
 * engine and the client, keeping this service easy to unit-test with a mock
 * {@code CvEngineClient}.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ImageServiceImpl implements ImageService {

    /**
     * MIME types accepted for forwarding to the CV engine.
     * Must match the {@code accepted_mime_types} list in the Python engine's
     * {@code config.py} to prevent forwarding files the engine would reject.
     */
    private static final Set<String> ACCEPTED_MIME_TYPES = Set.of(
            "image/jpeg",
            "image/png",
            "image/webp"
    );

    /** HTTP client for the Python/FastAPI CV engine. */
    private final CvEngineClient cvEngineClient;

    /**
     * {@inheritDoc}
     *
     * <p>Validates the file locally (fast, no I/O) before forwarding to the
     * CV engine (slow, network I/O) to avoid wasting bandwidth on obviously
     * invalid inputs.
     */
    @Override
    public EnhancedImageResult processUpload(MultipartFile file) {

        // --- Guard: empty file ---
        if (file.isEmpty()) {
            log.warn("Rejected upload — file is empty.");
            throw new InvalidImageException(
                    "Uploaded file is empty. Please provide a valid image."
            );
        }

        // --- Guard: MIME type ---
        String contentType = file.getContentType();
        if (contentType == null || !ACCEPTED_MIME_TYPES.contains(contentType.toLowerCase())) {
            log.warn("Rejected upload — unsupported content type '{}'.", contentType);
            throw new InvalidImageException(String.format(
                    "Unsupported image type '%s'. Accepted types are: %s",
                    contentType, ACCEPTED_MIME_TYPES
            ));
        }

        log.info(
                "Forwarding image to CV engine — file='{}', type={}, size={} B",
                file.getOriginalFilename(), contentType, file.getSize()
        );

        // --- Delegate to CV engine ---
        // CvEngineClient translates HTTP/connectivity errors into CvEngineException,
        // which is handled by GlobalExceptionHandler → 502/503 response to client.
        return cvEngineClient.enhance(file);
    }
}
