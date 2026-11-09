"""
Project NETRA — FastAPI ML Engine
===================================
Companion Python/FastAPI service to the Java Spring Boot backend.
Receives raw underwater images, runs the CV enhancement pipeline, and
streams the processed image bytes back to the caller.

Architecture
------------
  Java Spring Boot (port 8080)
       │  POST /api/v1/netra/enhance  (multipart image)
       ▼
  Python FastAPI Engine (port 8000)   ← this service
       │  POST /process-image         (multipart image)
       │
       ├── core/enhancement.py        ← Classical CV pipeline (Phase 1)
       │       ├── Gray World Color Correction
       │       └── CLAHE in LAB colour space
       │
       └── [stub] torch_enhance()     ← Deep-learning model (Phase 2)

Running
-------
    # Development (auto-reload)
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # Production
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from core.enhancement import run_pipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("netra.main")


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """
    Startup: log configuration summary and pre-warm any heavy resources.
    Shutdown: release model handles, flush buffers, etc.
    """
    logger.info("=" * 60)
    logger.info("🌊 Project NETRA CV Engine starting up")
    logger.info("   Version  : %s", settings.app_version)
    logger.info("   Debug    : %s", settings.debug)
    logger.info("   Max size : %.1f MB", settings.max_upload_size_mb)
    logger.info("   GrayWorld: %s", settings.gray_world_enabled)
    logger.info("   CLAHE    : %s (clip=%.1f, tiles=%dx%d)",
                settings.clahe_enabled,
                settings.clahe_clip_limit,
                settings.clahe_tile_grid_size,
                settings.clahe_tile_grid_size)
    logger.info("=" * 60)

    # TODO Phase 2: load PyTorch model weights here
    # global model
    # model = load_netra_model("weights/netra_v1.pth")
    # logger.info("Model loaded successfully.")

    yield  # <-- application runs

    logger.info("🌊 NETRA CV Engine shutting down — goodbye.")


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered underwater image enhancement engine for Project NETRA. "
        "Accepts raw underwater imagery and returns colour-corrected, "
        "contrast-enhanced output ready for deep-learning inference."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.java_backend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each incoming request with method, path, and elapsed time."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Helper — validation
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES: int = int(settings.max_upload_size_mb * 1024 * 1024)


async def _validate_upload(file: UploadFile) -> bytes:
    """
    Read and validate an uploaded image file.

    Checks
    ------
    1. MIME type is in the accepted list.
    2. File is not empty.
    3. File size does not exceed the configured limit.

    Returns
    -------
    Raw file bytes if all checks pass.

    Raises
    ------
    HTTPException (400 / 413) on any validation failure.
    """
    # --- MIME type ---
    if file.content_type not in settings.accepted_mime_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "UNSUPPORTED_FILE_TYPE",
                "message": (
                    f"Content type '{file.content_type}' is not supported. "
                    f"Accepted types: {settings.accepted_mime_types}"
                ),
            },
        )

    # --- Read bytes ---
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "EMPTY_FILE",
                "message": "The uploaded file is empty.",
            },
        )

    # --- Size guard ---
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "FILE_TOO_LARGE",
                "message": (
                    f"File size {len(raw_bytes):,} bytes exceeds the "
                    f"maximum allowed size of {MAX_UPLOAD_BYTES:,} bytes "
                    f"({settings.max_upload_size_mb:.0f} MB)."
                ),
            },
        )

    return raw_bytes


# ---------------------------------------------------------------------------
# Helper — encode result image
# ---------------------------------------------------------------------------

def _encode_image(bgr: np.ndarray) -> bytes:
    """
    Encode a BGR NumPy array back into compressed image bytes.

    Format and quality are driven by ``settings.output_format`` and
    ``settings.jpeg_quality``.

    Parameters
    ----------
    bgr:
        Processed image in OpenCV BGR format (uint8).

    Returns
    -------
    Compressed image bytes ready for streaming.

    Raises
    ------
    RuntimeError if OpenCV encoding fails (should not normally happen).
    """
    if settings.output_format.upper() == "PNG":
        ok, buffer = cv2.imencode(".png", bgr)
        media_type = "image/png"
    else:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
        ok, buffer = cv2.imencode(".jpg", bgr, encode_params)
        media_type = "image/jpeg"

    if not ok or buffer is None:
        raise RuntimeError("OpenCV failed to encode the processed image.")

    return bytes(buffer), media_type


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe", tags=["Ops"])
async def health_check() -> dict:
    """
    Returns the operational status of the CV engine.

    Intended for use as a Kubernetes/Docker liveness probe and for the
    Java Spring Boot backend to verify this service is reachable.
    """
    return {
        "status": "UP",
        "service": settings.app_name,
        "version": settings.app_version,
        "pipeline": {
            "gray_world": settings.gray_world_enabled,
            "clahe": settings.clahe_enabled,
            "torch_model": False,  # Phase 2 — not yet loaded
        },
    }


@app.post(
    "/process-image",
    summary="Enhance an underwater image",
    tags=["Enhancement"],
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}},
            "description": (
                "The enhanced image bytes, with NETRA pipeline metadata "
                "embedded in response headers."
            ),
        },
        400: {"description": "Invalid file type or empty upload."},
        413: {"description": "File exceeds the maximum allowed size."},
        422: {"description": "Image could not be decoded (corrupt or unsupported format)."},
        500: {"description": "Internal pipeline error."},
    },
)
async def process_image(
    image: UploadFile = File(
        ...,
        description=(
            "Raw underwater image to enhance. "
            "Accepted formats: JPEG, PNG, WebP. "
            f"Maximum size: {settings.max_upload_size_mb:.0f} MB."
        ),
    ),
) -> StreamingResponse:
    """
    **Core NETRA enhancement endpoint.**

    Accepts a multipart image upload, runs it through the full CV pipeline,
    and streams the enhanced image back to the caller.

    ### Processing Steps (Phase 1 — Classical CV)

    1. **Decode** — Convert raw bytes to a NumPy BGR array via OpenCV.
    2. **Gray World Color Correction** — Rescales each colour channel so the
       average reflectance equals the global mean, compensating for the
       blue-green colour cast caused by water's selective light absorption.
    3. **CLAHE** — Applies Contrast Limited Adaptive Histogram Equalisation
       in the LAB colour space (L channel only), boosting local contrast
       without amplifying noise or causing halo artefacts.

    ### Response Headers

    The response includes the following custom headers for observability:

    | Header | Description |
    |--------|-------------|
    | `X-NETRA-Steps` | Comma-separated list of pipeline steps executed |
    | `X-NETRA-Original-Filename` | Original filename from the upload |
    | `X-NETRA-Processing-Time-Ms` | End-to-end processing time in milliseconds |

    ### Example (`curl`)

    ```bash
    curl -X POST http://localhost:8000/process-image \\
      -F "image=@/path/to/underwater.jpg" \\
      --output enhanced.jpg
    ```
    """
    t_start = time.perf_counter()

    # --- 1. Validate upload ---
    raw_bytes = await _validate_upload(image)
    original_filename = image.filename or "unknown"
    logger.info(
        "POST /process-image — file='%s', size=%d B, type=%s",
        original_filename, len(raw_bytes), image.content_type,
    )

    # --- 2. Run enhancement pipeline ---
    try:
        result = run_pipeline(raw_bytes)
    except ValueError as exc:
        # Corrupt / undecodable image
        logger.warning("Image decode failed for '%s': %s", original_filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "IMAGE_DECODE_FAILED", "message": str(exc)},
        ) from exc
    except Exception as exc:  # noqa: BLE001
        # Unexpected pipeline error — log full traceback, return generic message
        logger.exception("Unexpected pipeline error for '%s'", original_filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PIPELINE_ERROR",
                "message": "An unexpected error occurred during image processing.",
            },
        ) from exc

    # --- 3. Encode result ---
    try:
        image_bytes, media_type = _encode_image(result.image_bgr)
    except RuntimeError as exc:
        logger.exception("Encoding failure for '%s'", original_filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "ENCODING_ERROR", "message": str(exc)},
        ) from exc

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        "Enhancement complete — '%s': %d B → %d B in %.1f ms",
        original_filename, len(raw_bytes), len(image_bytes), elapsed_ms,
    )

    # --- 4. Build streaming response with metadata headers ---
    headers = {
        "X-NETRA-Steps": ", ".join(result.steps_applied),
        "X-NETRA-Original-Filename": original_filename,
        "X-NETRA-Processing-Time-Ms": f"{elapsed_ms:.1f}",
        # Allow the Java caller to read these headers cross-origin
        "Access-Control-Expose-Headers": (
            "X-NETRA-Steps, X-NETRA-Original-Filename, X-NETRA-Processing-Time-Ms"
        ),
    }

    return StreamingResponse(
        content=io.BytesIO(image_bytes),
        media_type=media_type,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Custom exception handler — ensure all errors use a consistent JSON shape
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wraps FastAPI's default HTTP exception into the NETRA envelope format."""
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "ERROR",
            **detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler — prevents raw tracebacks leaking to clients."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "status": "ERROR",
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
