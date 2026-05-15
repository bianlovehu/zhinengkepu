"""
向量嵌入模型
"""
import asyncio
import logging
import time
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    向量嵌入模型

    支持多种Embedding Provider:
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - 本地模型 (sentence-transformers)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        provider: str = "openai",
        rate_limit: float = 0.5,  # 每次请求间隔（秒）
        rate_limit_retry_seconds: float = 65.0
    ):
        """
        Args:
            model_name: 模型名称
            dimension: 向量维度
            provider: 提供商 (openai / local)
            rate_limit: 请求间隔（秒），避免触发API限流
            rate_limit_retry_seconds: 触发 QPM/429 限流后的等待时间（秒）
        """
        self.model_name = model_name
        self.dimension = dimension
        self.provider = provider
        self.rate_limit = rate_limit
        self.rate_limit_retry_seconds = rate_limit_retry_seconds
        self._client = None
        self._last_request_time = 0.0

    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """创建嵌入客户端"""
        if self.provider == "openai":
            try:
                from openai import AsyncOpenAI
                from config import get_settings
                settings = get_settings()

                return AsyncOpenAI(
                    api_key=settings.EMBEDDING_API_KEY,
                    base_url=settings.EMBEDDING_BASE_URL or None
                )
            except ImportError:
                logger.error("OpenAI SDK not installed")
                return None

        elif self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("sentence-transformers not installed")
                return None

        return None

    async def embed(self, text: str) -> List[float]:
        """
        单条文本向量化

        Args:
            text: 输入文本

        Returns:
            List[float]: 向量
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: 向量列表
        """
        if not texts:
            return []

        # 速率控制：确保两次请求之间有足够间隔
        await self._rate_limit_wait()

        if self.client is None:
            raise RuntimeError("Embedding client not initialized")

        # API 请求失败时的重试配置（指数退避）
        max_retries = 8
        base_delay = 2.0  # 基础等待时间（秒）
        max_delay = 120.0  # 最大等待时间

        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    result = await self._embed_openai(texts)
                elif self.provider == "local":
                    result = self._embed_local(texts)
                else:
                    raise ValueError(f"Unsupported embedding provider: {self.provider}")

                return self._validate_embeddings(result)

            except Exception as e:
                # 计算指数退避时间
                delay = min(base_delay * (2 ** attempt), max_delay)
                if self._is_rate_limit_error(e):
                    delay = max(delay, self.rate_limit_retry_seconds)
                logger.warning(f"Batch embedding attempt {attempt + 1}/{max_retries} failed: {e}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(
                        f"All {max_retries} embedding attempts failed for {len(texts)} texts"
                    ) from e

    async def _rate_limit_wait(self):
        """速率控制：确保请求间隔不小于 rate_limit"""
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        if elapsed < self.rate_limit and elapsed >= 0:
            wait_time = self.rate_limit - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        self._last_request_time = time.time()

    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """OpenAI嵌入"""
        logger.info(f"Calling OpenAI embedding API for {len(texts)} texts, model: {self.model_name}")
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        logger.info(f"Embedding API returned {len(response.data)} results")
        return [item.embedding for item in response.data]

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """识别 OpenAI 兼容接口的 429/QPM 限流错误。"""
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True

        error_text = str(error).lower()
        rate_limit_markers = (
            "429",
            "ratelimit",
            "rate limit",
            "qpm",
            "请求过于频繁",
            "too many requests",
        )
        return any(marker in error_text for marker in rate_limit_markers)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """本地模型嵌入"""
        embeddings = self.client.encode(texts)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

    def _fallback_embed(self, texts: List[str]) -> List[List[float]]:
        """保留兼容入口，但正式链路禁止随机向量落库。"""
        raise RuntimeError("Random fallback embeddings are disabled")

    def _validate_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """验证嵌入向量维度。维度不匹配必须显式修配置或重建库。"""
        if not embeddings:
            return embeddings
        
        expected_dim = self.dimension
        validated = []
        for emb in embeddings:
            actual_dim = len(emb)
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                    "Update EMBEDDING_DIMENSION or use a matching embedding model, then rebuild the knowledge base."
                )
            validated.append(emb)
        return validated

    async def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 余弦相似度 [0, 1]
        """
        emb1, emb2 = await self.embed_batch([text1, text2])

        # 余弦相似度
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)
