from pydantic import Field
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "seamless-retail"
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: str = "http://localhost:5173"

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "retail"
    POSTGRES_USER: str = "retail"
    POSTGRES_PASSWORD: str = "retail"

    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "retail_context"

    OPENAI_API_KEY: str | None = None
    HUGGINGFACE_API_KEY: str | None = None
    HUGGINGFACE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
