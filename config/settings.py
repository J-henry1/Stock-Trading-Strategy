"""
Centralized configuration using Pydantic Settings.
Reads from .env file and environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # --- API Keys ---
    finnhub_api_key: str = Field(
            default="",
            description="Finnhub API key from https://finnhub.io"
        )

    # --- Server ---
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- Model ---
    model_dir: str = Field(default="models")
    model_version: str = Field(default="xgb_v1")
    history_days: int = Field(
        default=180,
        description="Number of calendar days of historical price data to fetch"
    )
    cv_folds: int = Field(
        default=5,
        description="Number of stratified cross-validation folds"
    )

    # --- CORS ---
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_version}.json")

    @property
    def scaler_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_version}_scaler.pkl")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.model_version}_metadata.json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra:"ignore"


# Singleton instance
settings = Settings()
