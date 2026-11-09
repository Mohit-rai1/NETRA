package com.Netra.app.client;

import com.Netra.app.dto.response.EnhancedImageResult;
import com.Netra.app.exception.CvEngineException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

/**
 * HTTP client for the NETRA Python/FastAPI CV engine.
 *
 * <p>Wraps a {@link RestClient} bean (pre-configured with the engine base URL)
 * and exposes a single method: {@link #enhance(MultipartFile)}.  All HTTP errors
 * and connectivity failures are translated into {@link CvEngineException} so
 * callers do not need to handle Spring's low-level HTTP exceptions.
 *
 * <h3>Request shape</h3>
 * <pre>
 *   POST /process-image
 *   Content-Type: multipart/form-data
 *
 *   --boundary
 *   Content-Disposition: form-data; name="image"; filename="...jpg"
 *   Content-Type: image/jpeg
 *
 *   &lt;raw image bytes&gt;
 *   --boundary--
 * </pre>
 *
 * <h3>Response shape</h3>
 * <pre>
 *   HTTP/1.1 200 OK
 *   Content-Type: image/jpeg
 *   X-NETRA-Steps: decode, color_balance(...), mpr(...), ...
 *   X-NETRA-Original-Filename: underwater.jpg
 *   X-NETRA-Processing-Time-Ms: 14.3
 *
 *   &lt;raw enhanced image bytes&gt;
 * </pre>
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class CvEngineClient {

    // Header names emitted by the Python engine — mirrored in ImageController
    public static final String HEADER_STEPS         = "X-NETRA-Steps";
    public static final String HEADER_FILENAME      = "X-NETRA-Original-Filename";
    public static final String HEADER_PROC_TIME_MS  = "X-NETRA-Processing-Time-Ms";

    private static final String PROCESS_IMAGE_PATH  = "/process-image";

    /** Pre-configured RestClient bean wired with the CV engine base URL. */
    private final RestClient cvEngineRestClient;

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Forwards a multipart image to the Python CV engine and returns the
     * enhanced image bytes along with pipeline metadata headers.
     *
     * <p>Processing steps performed by this method:
     * <ol>
     *   <li>Read the uploaded file bytes into a {@link ByteArrayResource}.</li>
     *   <li>Build a {@code multipart/form-data} request body.</li>
     *   <li>POST to {@code /process-image} and receive image bytes + headers.</li>
     *   <li>Map all HTTP and connectivity errors to {@link CvEngineException}.</li>
     * </ol>
     *
     * @param file the multipart image uploaded by the client
     * @return {@link EnhancedImageResult} with image bytes and metadata
     * @throws CvEngineException if the engine returns a non-2xx status or is unreachable
     * @throws CvEngineException (503) if the engine connection is refused / times out
     */
    public EnhancedImageResult enhance(MultipartFile file) {
        byte[] fileBytes = readFileBytes(file);

        // Build the multipart body with explicit Content-Disposition and Content-Type
        MultiValueMap<String, Object> form = buildMultipartBody(file, fileBytes);

        log.debug(
                "POST {} — file='{}', size={} B, type={}",
                PROCESS_IMAGE_PATH, file.getOriginalFilename(), fileBytes.length, file.getContentType()
        );

        try {
            ResponseEntity<byte[]> response = cvEngineRestClient.post()
                    .uri(PROCESS_IMAGE_PATH)
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(form)
                    .retrieve()
                    .toEntity(byte[].class);

            return extractResult(response, file);

        } catch (RestClientResponseException ex) {
            // Engine responded with 4xx / 5xx
            log.error(
                    "CV engine HTTP error: status={}, body={}",
                    ex.getStatusCode().value(),
                    ex.getResponseBodyAsString()
            );
            throw new CvEngineException(
                    ex.getStatusCode().value(),
                    "CV engine rejected the request: " + ex.getResponseBodyAsString(),
                    ex
            );

        } catch (ResourceAccessException ex) {
            // Connection refused, timeout, DNS failure, etc.
            log.error("CV engine unreachable: {}", ex.getMessage());
            throw new CvEngineException(
                    503,
                    "CV engine is not reachable. Ensure the Python service is running on the configured port.",
                    ex
            );
        }
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    /**
     * Reads all bytes from the uploaded file, wrapping {@link IOException}
     * in a {@link CvEngineException} so the caller chain stays exception-safe.
     */
    private byte[] readFileBytes(MultipartFile file) {
        try {
            return file.getBytes();
        } catch (IOException ex) {
            throw new CvEngineException(400, "Could not read uploaded file bytes: " + ex.getMessage(), ex);
        }
    }

    /**
     * Constructs the {@code multipart/form-data} body expected by the Python engine.
     *
     * <p>The part is named {@code "image"} with its original content-type and
     * filename preserved — the engine uses these for MIME validation and logging.
     */
    private MultiValueMap<String, Object> buildMultipartBody(MultipartFile file, byte[] fileBytes) {
        // Explicit Content-Type header for the image part
        HttpHeaders partHeaders = new HttpHeaders();
        partHeaders.setContentType(MediaType.parseMediaType(
                StringUtils.hasText(file.getContentType()) ? file.getContentType() : "image/jpeg"
        ));

        // ByteArrayResource with overridden getFilename() so Spring sets Content-Disposition correctly
        String filename = StringUtils.hasText(file.getOriginalFilename())
                ? file.getOriginalFilename() : "image.jpg";

        ByteArrayResource imageResource = new ByteArrayResource(fileBytes) {
            @Override
            public String getFilename() {
                return filename;
            }
        };

        MultiValueMap<String, Object> form = new LinkedMultiValueMap<>();
        form.add("image", new HttpEntity<>(imageResource, partHeaders));
        return form;
    }

    /**
     * Extracts the image bytes and NETRA pipeline metadata from the engine response.
     *
     * <p>Falls back gracefully if any header is absent (e.g. during development
     * with a simplified engine build).
     */
    private EnhancedImageResult extractResult(ResponseEntity<byte[]> response, MultipartFile file) {
        HttpHeaders headers = response.getHeaders();

        String contentType = headers.getContentType() != null
                ? headers.getContentType().toString() : "image/jpeg";

        String stepsApplied     = headers.getFirst(HEADER_STEPS);
        String originalFilename = headers.getFirst(HEADER_FILENAME);
        String processingTimeMs = headers.getFirst(HEADER_PROC_TIME_MS);

        // Fallback: use the originally uploaded filename if the engine header is absent
        if (!StringUtils.hasText(originalFilename)) {
            originalFilename = file.getOriginalFilename();
        }

        byte[] imageBytes = response.getBody();
        if (imageBytes == null || imageBytes.length == 0) {
            throw new CvEngineException(500, "CV engine returned an empty response body.");
        }

        log.info(
                "Enhancement complete — file='{}', output={} B, time={}ms, steps=[{}]",
                originalFilename, imageBytes.length, processingTimeMs, stepsApplied
        );

        return new EnhancedImageResult(imageBytes, contentType, originalFilename, stepsApplied, processingTimeMs);
    }
}
