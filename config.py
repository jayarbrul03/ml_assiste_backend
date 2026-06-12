from pathlib import Path

from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'avplatform.db'}"
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    db_reset_on_start: bool = False
    storage_backend: str = "local"  # local | minio
    local_storage_dir: Path = BASE_DIR / "data" / "uploads"
    local_files_url: str = "http://localhost:8000/files"
    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "av-assets"
    minio_secure: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    autolabel_enabled: bool = True
    autolabel_confidence_threshold: float = 0.85
    autolabel_auto_submit: bool = True
    active_learning_uncertainty_threshold: float = 0.3
    qa_iou_threshold: float = 0.5
    qa_pass_threshold: float = 0.7
    sla_hours: int = 24
    cors_origins: str = "http://localhost:3000"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"


settings = Settings()
settings.local_storage_dir.mkdir(parents=True, exist_ok=True)
(settings.local_storage_dir.parent).mkdir(parents=True, exist_ok=True)
