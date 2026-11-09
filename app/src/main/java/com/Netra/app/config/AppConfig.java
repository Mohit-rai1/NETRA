package com.Netra.app.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

import java.net.http.HttpClient;
import java.time.Duration;

/**
 * Spring application-level bean configuration for Project NETRA.
 *
 * <p>Registers a pre-configured {@link RestClient} bean pointed at the Python
 * CV engine. By externalising the base URL to {@code application.properties},
 * the client URL can be overridden per environment (dev / staging / prod) without
 * touching source code.
 *
 * <h3>Why HTTP/1.1 is forced explicitly</h3>
 * <p>Java 21's {@link java.net.http.HttpClient} defaults to {@code HTTP_2} preferred.
 * uvicorn (the Python ASGI server backing FastAPI) only speaks HTTP/1.1 in its
 * standard single-worker configuration; when it receives an HTTP/2 binary connection
 * preface it responds with <em>"Invalid HTTP request received."</em> and drops the
 * connection. Pinning to {@code HTTP_1_1} here eliminates the protocol mismatch entirely.
 */
@Configuration
public class AppConfig {

    /**
     * Base URL of the Python/FastAPI CV engine.
     * Injected from {@code netra.cv-engine.base-url} in {@code application.properties}.
     * Example: {@code http://localhost:8000}
     */
    @Value("${netra.cv-engine.base-url}")
    private String cvEngineBaseUrl;

    /**
     * Creates a {@link RestClient} pre-configured for the NETRA CV engine.
     *
     * <p>Configuration highlights:
     * <ul>
     *   <li><b>HTTP/1.1 only</b> — avoids the HTTP/2 ↔ uvicorn protocol mismatch.</li>
     *   <li><b>Connect timeout: 10 s</b> — surfaces "engine not running" failures fast
     *       so they map cleanly to 503 Service Unavailable.</li>
     *   <li><b>Read timeout: 60 s</b> — accommodates the Python ML pipeline on
     *       slower hardware without hitting a premature timeout during inference.</li>
     *   <li><b>Accept: * / *</b> — accepts both binary image and JSON error responses.</li>
     * </ul>
     *
     * @return configured {@code RestClient} instance ready for injection into
     *         {@link com.Netra.app.client.CvEngineClient}
     */
    @Bean
    public RestClient cvEngineRestClient() {

        // Pin to HTTP/1.1 — prevents Java 21 from sending HTTP/2 binary frames
        // to uvicorn, which only speaks HTTP/1.1 and would reject them with
        // "Invalid HTTP request received."
        HttpClient httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(httpClient);
        factory.setReadTimeout(Duration.ofSeconds(60)); // allow time for the ML pipeline

        return RestClient.builder()
                .baseUrl(cvEngineBaseUrl)
                .requestFactory(factory)
                .defaultHeader(HttpHeaders.ACCEPT, MediaType.ALL_VALUE)
                .defaultHeader(HttpHeaders.USER_AGENT, "NETRA-Java-Backend/1.0")
                .build();
    }
}
