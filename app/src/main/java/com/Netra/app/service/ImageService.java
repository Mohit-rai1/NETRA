package com.Netra.app.service;

import com.Netra.app.dto.response.EnhancedImageResult;
import org.springframework.web.multipart.MultipartFile;

/**
 * Service contract for NETRA image processing operations.
 *
 * <p>Accepts a raw multipart upload, validates it, forwards it to the Python
 * CV engine, and returns the fully-processed enhanced image together with
 * pipeline metadata.
 */
public interface ImageService {

    /**
     * Validates the uploaded image and delegates enhancement to the CV engine.
     *
     * <p>Processing steps performed by implementations:
     * <ol>
     *   <li>Guard against empty files.</li>
     *   <li>Validate MIME type against the accepted list.</li>
     *   <li>Forward the file to the Python/FastAPI CV engine via
     *       {@link com.Netra.app.client.CvEngineClient}.</li>
     *   <li>Return the enhanced image bytes and pipeline metadata.</li>
     * </ol>
     *
     * @param file the multipart image received from the client
     * @return {@link EnhancedImageResult} containing enhanced image bytes,
     *         content type, original filename, pipeline steps, and processing time
     * @throws com.Netra.app.exception.InvalidImageException if the file is
     *         empty or its MIME type is not supported
     * @throws com.Netra.app.exception.CvEngineException if the CV engine
     *         returns an error or is unreachable
     */
    EnhancedImageResult processUpload(MultipartFile file);
}
