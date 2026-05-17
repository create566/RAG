"""
BM25 关键词检索通道
基于rank_bm25算法的简单实现
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    chunk_id: str
    parent_id: str = ""
    document_id: str = ""
    score: float = 0.0
    channel: str = "keyword"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BM25:
    """BM25算法实现"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = {}  # term -> doc frequency
        self.N = 0  # total docs
        self.corpus = []  # list of tokenized documents

    def index(self, corpus: List[str]):
        """建立BM25索引"""
        self.corpus = [self._tokenize(doc) for doc in corpus]
        self.N = len(self.corpus)
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # 计算文档频率
        self.doc_freqs = {}
        for doc in self.corpus:
            seen_terms = set(doc)
            for term in seen_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def _tokenize(self, text: str) -> List[str]:
        """中文分词 - 使用jieba"""
        import re
        text = text.lower()

        # 尝试使用jieba分词
        try:
            import jieba
            # 精确模式分词
            tokens = list(jieba.cut(text, cut_all=False))
            return [t for t in tokens if len(t) > 1 and not t.isspace()]
        except ImportError:
            pass

        # 回退方案：简单分词 - 支持中英文
        chinese_pattern = re.compile(r'[一-鿿]+')
        english_pattern = re.compile(r'[a-zA-Z0-9]+')

        tokens = []
        # 提取中文字符串作为词
        for match in chinese_pattern.findall(text):
            tokens.append(match)
        # 提取英文/数字词
        for match in english_pattern.findall(text):
            if len(match) > 1:
                tokens.append(match.lower())

        return tokens

    def get_idf(self, term: str) -> float:
        """计算IDF"""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        """计算query对文档的BM25分数"""
        query_terms = self._tokenize(query)
        doc = self.corpus[doc_index]
        doc_len = self.doc_lengths[doc_index]

        score = 0.0
        for term in query_terms:
            if term not in self.doc_freqs:
                continue
            tf = doc.count(term)
            if tf == 0:
                continue
            idf = self.get_idf(term)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            score += idf * tf_norm
        return score


class KeywordRetrievalChannel:
    """关键词检索通道 - 使用内存BM25实现"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.bm25 = BM25(
            k1=self.config.get("bm25_k1", 1.5),
            b=self.config.get("bm25_b", 0.75)
        )
        self._indexed = False
        self._document_chunks = []  # (chunk_id, document_id, document_name, content)
        self._metadata_map = {}  # chunk_id -> metadata

    def index_documents(self, chunks: List[Dict]):
        """索引文档块"""
        self._document_chunks = []
        self._metadata_map = {}

        contents = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            doc_id = chunk.get("document_id", "")
            doc_name = chunk.get("document_name", "")
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            self._document_chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_name": doc_name,
                "content": content
            })
            self._metadata_map[chunk_id] = metadata
            contents.append(content)

        self.bm25.index(contents)
        self._indexed = True

    def retrieve(self, query: str, document_ids: List[str] = None, top_k: int = 10) -> List[RetrievalResult]:
        """检索相关文档"""
        if not self._indexed or not query:
            logger.info(f"[KEYWORD] 未索引或查询为空: indexed={self._indexed}, query={query}")
            return []

        logger.info(f"[KEYWORD] 查询: {query}, 文档数: {len(self._document_chunks)}")

        # 计算所有文档的BM25分数
        scores = []
        for i, chunk_info in enumerate(self._document_chunks):
            # 按document_ids过滤
            if document_ids:
                doc_id = str(chunk_info.get("document_id", ""))
                if doc_id not in document_ids:
                    continue

            score = self.bm25.score(query, i)
            if score > 0:
                scores.append((i, score))

        # 排序并取top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        min_score = self.config.get("min_keyword_score", 0.1)

        logger.info(f"[KEYWORD] BM25得分: {len(scores)} 个文档有得分, top: {scores[:3] if scores else []}")
        for i, score in scores[:top_k]:
            if score < min_score:
                continue
            chunk_info = self._document_chunks[i]
            chunk_id = chunk_info["chunk_id"]
            results.append(RetrievalResult(
                content=chunk_info["content"],
                chunk_id=chunk_id,
                parent_id=chunk_info.get("parent_id", ""),
                document_id=chunk_info.get("document_id", ""),
                score=score,
                channel="keyword",
                metadata=self._metadata_map.get(chunk_id, {})
            ))

        logger.info(f"[KEYWORD] 返回结果: {len(results)} 条")
        return results


class HybridRetrievalChannel:
    """混合检索通道 - 结合向量和关键词"""

    def __init__(self, vector_store, keyword_channel: KeywordRetrievalChannel = None, config: Dict = None):
        self.vector_store = vector_store
        self.keyword_channel = keyword_channel
        self.config = config or {}
        self.rrf_k = self.config.get("rrf_k", 60)

    def rrf_fuse(self, vector_results: List[Dict], keyword_results: List[Dict], top_k: int = 10) -> List[Dict]:
        """RRF融合"""
        scores = {}

        # 处理向量结果
        for rank, result in enumerate(sorted(vector_results, key=lambda x: x.get("score", 0), reverse=True)):
            chunk_id = result.get("chunk_id", "")
            if chunk_id not in scores:
                scores[chunk_id] = {"result": result, "rrf_score": 0}
            scores[chunk_id]["rrf_score"] += 1 / (self.rrf_k + rank + 1)

        # 处理关键词结果
        for rank, result in enumerate(sorted(keyword_results, key=lambda x: x.get("score", 0), reverse=True)):
            chunk_id = result.get("chunk_id", "")
            if chunk_id not in scores:
                scores[chunk_id] = {"result": result, "rrf_score": 0}
            scores[chunk_id]["rrf_score"] += 1 / (self.rrf_k + rank + 1)

        # 按RRF分数排序
        sorted_items = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        results = []
        for item in sorted_items[:top_k]:
            result = item["result"].copy()
            result["score"] = item["rrf_score"]
            result["channel"] = "hybrid"
            results.append(result)

        return results

    async def retrieve(self, query: str, embedding: List[float] = None, document_ids: List[str] = None, top_k: int = 10) -> List[Dict]:
        """执行混合检索"""
        vector_results = []
        keyword_results = []

        # 向量检索
        if self.vector_store and embedding:
            try:
                # 注意：Chroma的where过滤可能有兼容性问题，先不做过滤
                results = self.vector_store.query(
                    query_embeddings=[embedding],
                    n_results=top_k * 2
                )
                all_docs = results.get("documents", [[]])[0] if results.get("documents") else []
                all_metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
                all_dists = results.get("distances", [[]])[0] if results.get("distances") else []

                for i, doc in enumerate(all_docs):
                    if i >= len(all_metas):
                        continue
                    meta = all_metas[i] if isinstance(all_metas[i], dict) else {}
                    vector_results.append({
                        "content": doc,
                        "chunk_id": meta.get("chunk_id", ""),
                        "document_id": meta.get("document_id", ""),
                        "document_name": meta.get("document_name", ""),
                        "section_path": meta.get("section_path", ""),
                        "score": 1.0 - all_dists[i] if i < len(all_dists) else 0.0,
                        "channel": "vector",
                        "metadata": meta
                    })
            except Exception as e:
                logger.info(f"向量检索失败: {e}")

        # 关键词检索
        if self.keyword_channel:
            try:
                keyword_results = self.keyword_channel.retrieve(
                    query=query,
                    document_ids=document_ids,
                    top_k=top_k * 2
                )
            except Exception as e:
                logger.info(f"关键词检索失败: {e}")

        # RRF融合
        if vector_results or keyword_results:
            return self.rrf_fuse(vector_results, keyword_results, top_k)

        return vector_results or keyword_results