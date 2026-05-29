"""
Embedding 服务
统一封装文本向量化的调用
"""
from typing import List
import asyncio

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """文本向量化服务"""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._embedding_failure_count = 0
        self.embedding_available = True

    async def embed(self, text: str) -> List[float]:
        """获取单条文本的向量"""
        if not self.llm_service:
            return []

        try:
            emb = await self.llm_service.embed(text)
            if emb and len(emb) > 0:
                self._embedding_failure_count = 0
                self.embedding_available = True
                return emb
            return []
        except Exception as e:
            logger.warning(f"[Embedding] embed failed: {e}")
            self._embedding_failure_count += 1
            if self._embedding_failure_count >= 3:
                self.embedding_available = False
            return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本向量"""
        if not texts:
            return []

        results = []
        for text in texts:
            emb = await self.embed(text)
            results.append(emb if emb else [0] * 1536)  # 默认维度
        return results


def create_embedding_service(llm_service=None) -> EmbeddingService:
    """工厂方法创建 Embedding 服务"""
    return EmbeddingService(llm_service=llm_service)