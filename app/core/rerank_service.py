"""
SiliconFlow Rerank 服务
"""
from typing import List, Dict, Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class SiliconFlowRerankService:
    """SiliconFlow Rerank 服务"""

    def __init__(self, api_key: str, model: str = "BAAI/bge-reranker-base", min_score: float = 0.3):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.siliconflow.cn/v1/rerank"
        self.min_score = min_score  # 最低相关分阈值，低于此分的结果会被过滤

    async def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        对检索结果进行重排序

        Args:
            query: 查询问题
            results: 检索结果列表
            top_k: 返回前N个结果

        Returns:
            重排序后的结果
        """
        if not results:
            return []

        if not self.api_key:
            return results[:top_k]

        try:
            import httpx

            # 构建文档列表
            documents = []
            for r in results:
                # 兼容 RetrievalResult (dataclass) 和 dict
                if hasattr(r, 'content'):
                    documents.append(r.content)
                else:
                    documents.append(r.get("content", ""))

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_k
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    rerank_results = data.get("results", [])

                    # 按 rerank 分数重新排序，并过滤低分结果
                    reranked = []
                    for item in rerank_results:
                        idx = item.get("index", 0)
                        relevance_score = item.get("relevance_score", 0)
                        # 过滤掉低于阈值的低相关结果
                        if relevance_score < self.min_score:
                            logger.info(f"[RERANK] Filtered result (score={relevance_score:.3f} < {self.min_score})")
                            continue
                        if idx < len(results):
                            r = results[idx]
                            # 兼容 RetrievalResult (dataclass) 和 dict
                            if hasattr(r, 'content'):
                                from dataclasses import asdict
                                r = asdict(r)
                            r["rerank_score"] = relevance_score
                            reranked.append(r)

                    logger.info(f"[RERANK] Returned {len(reranked)}/{len(results)} results (min_score={self.min_score})")
                    return reranked
                else:
                    logger.warning(f"[RERANK] 请求失败: {response.status_code}")
                    # 失败时也转换为 dict，保持类型一致
                    reranked = []
                    for r in results[:top_k]:
                        if hasattr(r, 'content'):
                            from dataclasses import asdict
                            reranked.append(asdict(r))
                        else:
                            reranked.append(r)
                    return reranked

        except Exception as e:
            logger.warning(f"[RERANK] 错误: {e}")
            return results[:top_k]


class VLLMRerankService:
    """VLLM 本地 Rerank 服务"""

    def __init__(self, model: str = "BAAI/bge-reranker-base", base_url: str = "http://localhost:8012/v1", min_score: float = 0.3):
        self.model = model
        self.base_url = base_url
        self.min_score = min_score

    async def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        if not results:
            return []

        try:
            from openai import OpenAI
            client = OpenAI(api_key="vllm", base_url=self.base_url)

            documents = []
            for r in results:
                if hasattr(r, 'content'):
                    documents.append(r.content)
                else:
                    documents.append(r.get("content", ""))

            def _call():
                return client.moderations.create(
                    input=documents
                )

            # VLLM rerank 需要使用 /rerank 端点
            response = client.post(
                url=f"{self.base_url}/rerank",
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k
                }
            )

            if response.status_code == 200:
                data = response.json()
                rerank_results = data.get("results", [])

                reranked = []
                for item in rerank_results:
                    idx = item.get("index", 0)
                    relevance_score = item.get("relevance_score", 0)
                    if relevance_score < self.min_score:
                        continue
                    if idx < len(results):
                        r = results[idx]
                        if hasattr(r, 'content'):
                            from dataclasses import asdict
                            r = asdict(r)
                        r["rerank_score"] = relevance_score
                        reranked.append(r)
                return reranked
        except Exception as e:
            logger.warning(f"[VLLM RERANK] 错误: {e}")
        return results[:top_k]


class MockRerankService:
    """模拟 Rerank 服务（用于测试）"""

    async def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """直接返回原结果"""
        return results[:top_k]