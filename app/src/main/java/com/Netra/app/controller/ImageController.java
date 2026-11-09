package com.Netra.app.controller;

import com.Netra.app.client.CvEngineClient;
import com.Netra.app.dto.response.ApiResponse;
import com.Netra.app.dto.response.EnhancedImageResult;
import com.Netra.app.service.ImageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * REST controller for Project NETRA image enhancement operations.
 *
 * <p>All routes are versioned under {@code /api/v1/netra}.  This controller
 * is intentionally thin — it handles HTTP concerns only (binding, header
 * propagation, response shaping) and delegates all logic to {@link ImageService}.
 *
 * <h3>Enhanced endpoint response design</h3>
 * <p>The {@code /enhance} endpoint now returns the processed image bytes
 * directly (not a JSON acknowledgment), because:
 * <ul>
 *   <li>Clients can immediately display or forward the enhanced image.</li>
 *   <li>No polling or second round-trip is needed.</li>
 *   <li>Base64 encoding (which would be needed to embed bytes in JSON) adds
 *       ~33% size overhead.</li>
 * </ul>
 * Error responses are still JSON via {@link com.Netra.app.exception.GlobalExceptionHandler}.
 * Pipeline metadata is surfaced through {@code X-NETRA-*} custom headers.
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/netra")
@RequiredArgsConstructor
public class ImageController {

    private final ImageService imageService;

    // -------------------------------------------------------------------------
    // Core enhancement endpoint
    // -------------------------------------------------------------------------

    /**
     * Accepts a multipart underwater image, runs the full NETRA CV pipeline,
     * and streams the enhanced image back to the caller.
     *
     * <h3>Route</h3>
     * {@code POST /api/v1/netra/enhance}
     *
     * <h3>Consumes</h3>
     * {@code multipart/form-data} — part name: {@code image}
     *
     * <h3>Produces</h3>
     * {@code image/jpeg} or {@code image/png} bytes (determined by the CV engine).
     *
     * <h3>Response Headers</h3>
     * <table border="1">
     *   <tr><th>Header</th><th>Description</th></tr>
     *   <tr><td>{@code X-NETRA-Steps}</td>
     *       <td>Comma-separated pipeline steps executed (from Python engine)</td></tr>
     *   <tr><td>{@code X-NETRA-Original-Filename}</td>
     *       <td>Uploaded filename as received</td></tr>
     *   <tr><td>{@code X-NETRA-Processing-Time-Ms}</td>
     *       <td>Python-side processing time in milliseconds</td></tr>
     * </table>
     *
     * <h3>Example ({@code curl})</h3>
     * <pre>{@code
     * curl -X POST http://localhost:8080/api/v1/netra/enhance \
     *   -F "image=@underwater.jpg" \
     *   --output enhanced.jpg -v
     * }</pre>
     *
     * @param image the multipart image part from the client
     * @return {@code 200 OK} with enhanced image bytes and {@code X-NETRA-*} headers;
     *         error responses are JSON from {@link com.Netra.app.exception.GlobalExceptionHandler}
     */
    @PostMapping(
            value = "/enhance",
            consumes = MediaType.MULTIPART_FORM_DATA_VALUE,
            produces = {
                    MediaType.IMAGE_JPEG_VALUE,
                    MediaType.IMAGE_PNG_VALUE,
                    "image/webp",
                    MediaType.APPLICATION_JSON_VALUE  // allows JSON error responses to pass through
            }
    )
    public ResponseEntity<byte[]> enhance(
            @RequestPart("image") MultipartFile image
    ) {
        log.info(
                "POST /api/v1/netra/enhance — file='{}', size={} B, type={}",
                image.getOriginalFilename(), image.getSize(), image.getContentType()
        );

        EnhancedImageResult result = imageService.processUpload(image);

        HttpHeaders responseHeaders = buildNetraHeaders(result);

        return ResponseEntity
                .status(HttpStatus.OK)
                .headers(responseHeaders)
                .contentType(MediaType.parseMediaType(result.contentType()))
                .body(result.imageBytes());
    }

    // -------------------------------------------------------------------------
    // Health probe
    // -------------------------------------------------------------------------

    /**
     * Liveness check for the NETRA image pipeline (Java layer only).
     *
     * <p><b>Route:</b> {@code GET /api/v1/netra/health}
     *
     * <p>For a full end-to-end liveness check including the Python engine,
     * call {@code GET http://localhost:8000/health} directly.
     *
     * @return {@code 200 OK} with pipeline status
     */
    @GetMapping(value = "/health", produces = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<ApiResponse<String>> health() {
        return ResponseEntity.ok(
                ApiResponse.success("NETRA Java backend is operational.", "UP")
        );
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Constructs the {@code X-NETRA-*} response headers from the CV engine result.
     *
     * <p>These headers let upstream callers (browser devtools, monitoring dashboards,
     * API gateways) inspect the pipeline without parsing the binary image body.
     */
    private HttpHeaders buildNetraHeaders(EnhancedImageResult result) {
        HttpHeaders headers = new HttpHeaders();

        if (result.stepsApplied() != null) {
            headers.set(CvEngineClient.HEADER_STEPS, result.stepsApplied());
        }
        if (result.originalFilename() != null) {
            headers.set(CvEngineClient.HEADER_FILENAME, result.originalFilename());
        }
        if (result.processingTimeMs() != null) {
            headers.set(CvEngineClient.HEADER_PROC_TIME_MS, result.processingTimeMs());
        }
        // Expose the custom headers for cross-origin callers
        headers.set(
                "Access-Control-Expose-Headers",
                String.join(", ",
                        CvEngineClient.HEADER_STEPS,
                        CvEngineClient.HEADER_FILENAME,
                        CvEngineClient.HEADER_PROC_TIME_MS
                )
        );
        return headers;
    }
}
