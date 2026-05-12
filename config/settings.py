"""
项目配置管理
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """项目配置"""

    # ============ 项目路径 ============
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()
    DATA_DIR: Path = BASE_DIR / "knowledge_base"
    RAW_DOCS_DIR: Path = DATA_DIR / "raw_documents"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"

    # ============ API配置 ============
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TIMEOUT_TEXT: int = 20
    API_TIMEOUT_MULTIMODAL: int = 30

    # ============ 认证配置 ============
    API_TOKEN: str = "sk_customer_20260304"

    # ============ LLM配置 ============
    LLM_PROVIDER: str = "openai"  # openai / zhipu / dashscope
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: Optional[str] = None
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    # ============ 多模态模型配置 ============
    VISION_MODEL: str = "gpt-4o-mini"
    VISION_API_KEY: str = ""
    VISION_BASE_URL: Optional[str] = None

    # ============ Embedding配置 ============
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_DIMENSION: int = 1536

    # ============ RAG配置 ============
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5
    EMBEDDING_BATCH_SIZE: int = 50  # Embedding批量大小（越大越快，但需注意API限制）

    # ============ 向量数据库配置 ============
    VECTOR_DB_TYPE: str = "chroma"  # chroma / milvus / qdrant
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "knowledge_base" / "chroma_db")

    # ============ 会话配置 ============
    SESSION_EXPIRE_SECONDS: int = 3600
    MAX_SESSION_HISTORY: int = 20

    # ============ CORS配置 ============
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """获取单例配置"""
    return Settings()
