"""
Project NETRA — ML Engine Configuration
========================================
Centralised settings loaded from environment variables (or .env file).
All values have sensible defaults for local development.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings for the NETRA CV engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Server ---
    app_name: str = "NETRA CV Engine"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Upload Constraints ---
    max_upload_size_mb: float = 10.0          # Max raw file size in MB
    accepted_mime_types: list[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
    ]

    # --- Enhancement Pipeline ---
    # Gray World correction
    gray_world_enabled: bool = True

    # CLAHE (contrast enhancement)
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8             # (N x N) tiles

    # Output encoding
    output_format: str = "JPEG"               # JPEG | PNG
    jpeg_quality: int = 92                    # 0-100; 92 is perceptually lossless

    # --- Java Backend (caller) ---
    java_backend_url: str = "http://localhost:8080"


# Singleton settings instance — import this everywhere.
settings = Settings()
