"""
检索管道模块
包含：双通道混合检索、RRF融合、Parent-Child块聚合、证据预算控制
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    chunk_id: str
    parent_id: str
    document_id: str
    score: float
    channel: str  # "vector", "keyword", "hybrid"
    metadata: Dict[str, Any]


@dataclass
class SubQuestionEvidence:
    """子问题证据"""
    sub_question_index: int
    sub_question: str
    references: List[Dict[str, Any]]
    documents: List[str]
    fused_candidate_count: int = 0
    parent_candidate_count: int = 0
    reranked_candidate_count: int = 0
    channel_traces: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.channel_traces is None:
            self.channel_traces = []


@dataclass
class RagRetrievalContext:
    """RAG检索上下文"""
    retrieval_question: str
    sub_question_evidence_list: List[SubQuestionEvidence]
    used_channels: List[str]
    retrieval_notes: List[str]
    all_references: List[Dict[str, Any]]

    def is_empty(self) -> bool:
        return not self.all_references

    def flatten_references(self) -> List[Dict[str, Any]]:
        return self.all_references


class VectorRetrievalChannel:
    """向量检索通道"""

    def __init__(self, embedding_service, vector_store, config: Dict):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.config = config
        logger.info(f"[RETRIEVAL] 向量检索通道初始化成功")

    async def retrieve(self, query: str, document_ids: List[int] = None, top_k: int = 10, user_id: int = None) -> List[RetrievalResult]:
        """执行向量检索"""
        # 检查 embedding 是否可用
        if hasattr(self.embedding_service, 'embedding_available') and not self.embedding_service.embedding_available:
            logger.warning(f"[VECTOR] embedding_available=False，向量检索已禁用")
            return []

        # 获取查询向量
        logger.info(f"[VECTOR] 开始生成查询向量: query={query[:50]}")
        query_vector = await self.embedding_service.embed(query)
        if not query_vector:
            logger.warning(f"[VECTOR] 查询向量生成失败")
            return []

        logger.info(f"[VECTOR] 查询向量生成成功: dim={len(query_vector)}")

        # 构建过滤条件
        # 注意: Chroma 中存储的 document_id 是 UUID 字符串 (如 "b43125bc-9555-43ac-a92c-c56ddb1370e9")
        # 而 routed_document_id 可能是 int(doc_id[:8], 16) 转换的整数 (如 3023119804)
        # 这种情况下, 整数的 where 条件无法匹配 UUID 字符串, 所以只有当 document_ids
        # 包含看起来像 UUID 的字符串时才添加过滤
        where_filter = {}
        if document_ids and len(document_ids) > 0:
            # 检查是否包含 UUID 格式的字符串 (真正的 document_id)
            # UUID 格式: 8-4-4-4-12 共 36 个十六进制字符
            uuid_ids = [str(d) for d in document_ids if isinstance(d, str) and len(d) == 36 and "-" in d]
            if uuid_ids:
                where_filter["document_id"] = {"$in": uuid_ids}
            else:
                # 只有整数 ID 或空列表, 说明是通过 int(doc_id[:8], 16) 转换来的
                # 这种格式无法匹配 Chroma 中的 UUID 字符串, 跳过过滤以查询所有文档
                logger.info(f"[VECTOR] document_ids contains no UUID strings (only ints or empty) - querying all documents")
                where_filter = {}

        # 添加用户过滤（多用户隔离）
        if user_id is not None:
            where_filter["user_id"] = user_id

        # ChromaDB 0.5 只允许一个顶层操作符，多条件需用 $and 包裹
        if len(where_filter) > 1:
            where_filter = {"$and": [{k: v} for k, v in where_filter.items()]}

        # 从向量存储检索
        results = self.vector_store.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter if where_filter else None
        )

        if results is None:
            return []

        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        if not documents or not documents[0]:
            return []

        retrieval_results = []
        for i, doc in enumerate(documents[0]):
            metadata = metadatas[0][i] if metadatas and metadatas[0] else {}
            distance = distances[0][i] if distances and distances[0] else 1.0

            # 将距离转换为相似度
            # Chroma 使用 squared L2 distance，值可能很大(如 3000+)
            # 使用指数衰减公式: sim = exp(-distance / normalizer)
            # normalizer=5000 时: distance=3000→sim≈0.55, distance=10000→sim≈0.14
            if distance <= 1:
                similarity = 1 - distance
            else:
                normalizer = 5000
                similarity = __import__('math').exp(-distance / normalizer)

            if similarity >= self.config.get("min_vector_similarity", 0.08):
                retrieval_results.append(RetrievalResult(
                    content=doc,
                    chunk_id=metadata.get("chunk_id", ""),
                    parent_id=metadata.get("parent_id", ""),
                    document_id=metadata.get("document_id", ""),
                    score=similarity,
                    channel="vector",
                    metadata=metadata
                ))

        logger.info(f"[VECTOR] Chroma 检索完成: 共 {len(documents[0]) if documents and documents[0] else 0} 条, 通过阈值 {self.config.get('min_vector_similarity', 0.08)} 筛选后剩 {len(retrieval_results)} 条")
        return retrieval_results


class KeywordRetrievalChannel:
    """关键词检索通道"""

    def __init__(self, elasticsearch_client, config: Dict):
        self.es_client = elasticsearch_client
        self.config = config
        logger.info(f"[RETRIEVAL] 关键词检索通道初始化成功 | index={config.get('index')}")

    async def retrieve(self, query: str, document_ids: List[int] = None, top_k: int = 10, user_id: int = None) -> List[RetrievalResult]:
        """执行关键词检索"""
        # 构建查询
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "title", "tags"],
                                "type": "best_fields"
                            }
                        }
                    ]
                }
            },
            "size": top_k
        }

        # 添加文档过滤 - 只有当 document_ids 包含 UUID 格式字符串时才添加过滤
        # 注意: ES 中存储的 document_id 是 UUID 字符串, routed_document_id 可能是整数
        # 整数的过滤条件无法匹配 UUID 字符串, 所以跳过过滤以查询所有文档
        if document_ids and len(document_ids) > 0:
            uuid_ids = [str(d) for d in document_ids if isinstance(d, str) and len(d) == 36 and "-" in d]
            if uuid_ids:
                search_body["query"]["bool"]["filter"] = [
                    {"terms": {"document_id": uuid_ids}}
                ]
                logger.info(f"[KEYWORD] Filter by UUIDs: {uuid_ids}")
            else:
                logger.info(f"[KEYWORD] document_ids contains no UUID strings - querying all documents, document_ids={document_ids}")

        # 添加用户过滤
        if user_id is not None:
            if "filter" not in search_body["query"]["bool"]:
                search_body["query"]["bool"]["filter"] = []
            search_body["query"]["bool"]["filter"].append({"term": {"user_id": user_id}})

        try:
            def _search():
                return self.es_client.search(
                    index=self.config.get("index", "super_agent_keywords"),
                    body=search_body
                )
            response = await asyncio.to_thread(_search)

            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"] / 100  # 归一化
                if score >= self.config.get("min_keyword_score", 0.05):
                    results.append(RetrievalResult(
                        content=hit["_source"].get("content", ""),
                        chunk_id=hit["_source"].get("chunk_id", ""),
                        parent_id=hit["_source"].get("parent_id", ""),
                        document_id=hit["_source"].get("document_id", ""),
                        score=score,
                        channel="keyword",
                        metadata=hit["_source"]
                    ))
            return results
        except Exception as e:
            return []


class RRFer:
    """RRF(Reciprocal Rank Fusion)融合器 - 对标Java的RRF融合"""

    def __init__(self, k: int = 60):
        self.k = k
        logger.info(f"[RETRIEVAL] RRF融合器初始化成功 | k={k}")

    def fuse(self, vector_results: List[RetrievalResult], keyword_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """执行RRF融合"""
        scores = {}

        # 处理向量结果
        for rank, result in enumerate(sorted(vector_results, key=lambda x: x.score, reverse=True)):
            key = result.chunk_id
            if key not in scores:
                scores[key] = {"result": result, "rrf_score": 0}
            scores[key]["rrf_score"] += 1 / (self.k + rank + 1)

        # 处理关键词结果
        for rank, result in enumerate(sorted(keyword_results, key=lambda x: x.score, reverse=True)):
            key = result.chunk_id
            if key not in scores:
                scores[key] = {"result": result, "rrf_score": 0}
            scores[key]["rrf_score"] += 1 / (self.k + rank + 1)

        # 按RRF分数排序
        sorted_items = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

        fused_results = []
        for item in sorted_items:
            result = item["result"]
            result.channel = "hybrid"
            result.score = item["rrf_score"]
            fused_results.append(result)

        return fused_results


class ParentChildAggregator:
    """Parent-Child块聚合器 - 对标Java的聚合逻辑"""

    def __init__(self, vector_store, config: Dict):
        self.vector_store = vector_store
        self.config = config
        logger.info(f"[RETRIEVAL] ParentChild聚合器初始化成功 | max_parent_chunks={config.get('max_parent_chunks', 5)}")

    def aggregate(self, child_results: List[RetrievalResult], max_parent_chunks: int = 5) -> List[RetrievalResult]:
        """将Child块聚合到Parent块"""
        parent_ids = list(set([r.parent_id for r in child_results if r.parent_id]))[:max_parent_chunks]

        if not parent_ids:
            return child_results

        parent_results = []
        for pid in parent_ids:
            results = self.vector_store.get(where={"chunk_id": pid})
            if results is None:
                continue
            documents = results.get("documents")
            if not documents or not documents[0]:
                continue
            metadata = results["metadatas"][0] if results.get("metadatas") else {}
            parent_results.append(RetrievalResult(
                content=documents[0],
                chunk_id=pid,
                parent_id=pid,
                document_id=metadata.get("document_id", ""),
                score=child_results[0].score,
                channel="parent",
                metadata=metadata
            ))

        return parent_results if parent_results else child_results


class EvidenceBudgetController:
    """证据预算控制器 - 对标Java的证据预算控制"""

    def __init__(self, budget_per_child: int = 1500, budget_total: int = 4000,
                 llm_service=None, enable_semantic_compress: bool = False):
        self.budget_per_child = budget_per_child
        self.budget_total = budget_total
        self.semantic_compressor = SemanticCompressor(llm_service) if enable_semantic_compress and llm_service else None
        logger.info(f"[RETRIEVAL] 证据预算控制器初始化成功 | budget_total={budget_total}, semantic_compress={enable_semantic_compress}")

    def control(self, results: List[RetrievalResult]) -> Tuple[List[RetrievalResult], bool]:
        """控制证据预算（同步版本，使用句子边界截断）"""
        total_chars = sum(len(r.content if hasattr(r, 'content') else r.get("content", "") if isinstance(r, dict) else "") for r in results)

        if total_chars <= self.budget_total:
            return results, False

        trimmed = []
        char_count = 0
        for result in results:
            content = result.content if hasattr(result, 'content') else result.get("content", "") if isinstance(result, dict) else ""
            if char_count + len(content) <= self.budget_total:
                trimmed.append(result)
                char_count += len(content)
            else:
                # 在自然句子边界截断最后一个结果
                remaining = self.budget_total - char_count
                if remaining > 0:
                    truncated_content = self._truncate_at_sentence_boundary(content, remaining)
                    # 创建截断后的结果（保留原 result 的其他属性）
                    if hasattr(result, 'content'):
                        result.content = truncated_content
                    elif isinstance(result, dict):
                        result["content"] = truncated_content
                    trimmed.append(result)
                break

        return trimmed, True

    async def control_async(self, results: List[RetrievalResult], query: str = "") -> Tuple[List[RetrievalResult], bool]:
        """控制证据预算（异步版本，支持语义压缩）"""
        total_chars = sum(len(r.content if hasattr(r, 'content') else r.get("content", "") if isinstance(r, dict) else "") for r in results)

        if total_chars <= self.budget_total:
            return results, False

        # 需要压缩时，优先尝试语义压缩
        if self.semantic_compressor and total_chars > self.budget_total * 1.5:
            try:
                all_content = "\n\n".join(
                    r.content if hasattr(r, 'content') else r.get("content", "") if isinstance(r, dict) else ""
                    for r in results
                )
                compressed = await self.semantic_compressor.compress(query, all_content, self.budget_total)
                # 用压缩后的内容替换第一个结果
                if hasattr(results[0], 'content'):
                    results[0].content = compressed
                elif isinstance(results[0], dict):
                    results[0]["content"] = compressed
                return results[:1], True
            except Exception as e:
                logger.info(f"[BUDGET] Semantic compression failed, fallback to truncation: {e}")

        # 回退到句子边界截断
        return self.control(results), True

    def _truncate_at_sentence_boundary(self, content: str, max_len: int) -> str:
        """在自然句子边界截断内容"""
        if len(content) <= max_len or max_len <= 0:
            return content

        truncated = content[:max_len]
        # 查找中英文句号、问号、感叹号、换行符等自然断点
        punct_positions = [
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('.\n'),
            truncated.rfind('?\n'),
            truncated.rfind('!\n'),
            truncated.rfind('\n')
        ]
        # 取最靠后的有效位置
        valid_positions = [p for p in punct_positions if p > max_len * 0.3]
        if valid_positions:
            last_punct = max(valid_positions)
            return truncated[:last_punct + 1]
        return truncated + "..."


class SemanticCompressor:
    """语义压缩器 - 用 LLM 压缩上下文"""

    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def compress(self, query: str, context: str, max_length: int = 1000) -> str:
        """用 LLM 压缩上下文，保留与问题相关的内容"""
        prompt = f"""问题: {query}

上下文:
{context}

请提取与问题最相关的关键信息，删除冗余内容，保持语义完整。压缩后的内容不超过 {max_length} 字。"""

        try:
            compressed = await self.llm_service.chat(prompt)
            if compressed and len(compressed) <= max_length * 1.1:
                return compressed
            # 如果压缩结果仍然太长，再次截断
            return compressed[:max_length] + "..." if compressed else context[:max_length] + "..."
        except Exception as e:
            logger.info(f"[SEMANTIC COMPRESS] Error: {e}")
            # 失败时回退到原内容截断
            return context[:max_length] + "..."


class RagRetrievalEngine:
    """RAG检索引擎 - 对标Java的RagRetrievalEngine"""

    def __init__(self,
                 vector_channel: VectorRetrievalChannel,
                 keyword_channel: KeywordRetrievalChannel,
                 rerank_service=None,
                 llm_service=None,
                 config: Dict = None):
        self.vector_channel = vector_channel
        self.keyword_channel = keyword_channel
        self.rerank_service = rerank_service
        self.llm_service = llm_service
        self.config = config or {}

        self.rrfer = RRFer(k=self.config.get("rrf_k", 60))
        self.aggregator = ParentChildAggregator(vector_channel.vector_store, self.config)
        self.budget_controller = EvidenceBudgetController(
            budget_per_child=self.config.get("evidence_budget_per_child", 1500),
            budget_total=self.config.get("evidence_budget_total", 4000),
            llm_service=llm_service,
            enable_semantic_compress=self.config.get("enable_semantic_compress", False)
        )

    async def retrieve(self, plan, trace_recorder=None, user_id: int = None) -> RagRetrievalContext:
        """执行完整的检索流程"""
        retrieval_question = plan.retrieval_question
        sub_questions = plan.retrieval_sub_questions
        document_ids = plan.retrieval_document_ids

        all_references = []
        used_channels = set()
        retrieval_notes = []
        sub_question_evidence_list = []

        for idx, sub_q in enumerate(sub_questions):
            # 双通道并行检索
            # vector_channel 是 async 方法，直接调用
            vector_task = self.vector_channel.retrieve(sub_q, document_ids, self.config.get("vector_top_k", 10), user_id)

            # keyword_channel 可能是 BM25(同步) 或 ES(异步)，统一用 to_thread 包装
            if hasattr(self.keyword_channel, 'es_client'):
                # ES 版本是 async，直接调用
                keyword_task = self.keyword_channel.retrieve(sub_q, document_ids, self.config.get("keyword_top_k", 10), user_id)
            else:
                # BM25 版本是同步，用 to_thread 包装
                keyword_task = asyncio.to_thread(
                    self.keyword_channel.retrieve, sub_q, document_ids, self.config.get("keyword_top_k", 10), user_id
                )

            vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

            logger.info(f"[RETRIEVAL] sub_q={sub_q}, vector_results={len(vector_results)}, keyword_results={len(keyword_results)}")

            if vector_results:
                used_channels.add("vector")
            if keyword_results:
                used_channels.add("keyword")

            # RRF融合
            fused_results = self.rrfer.fuse(vector_results, keyword_results)

            # Parent-Child聚合
            if self.config.get("max_parent_chunks", 5) > 0:
                aggregated_results = self.aggregator.aggregate(fused_results, self.config.get("max_parent_chunks", 5))
            else:
                aggregated_results = fused_results

            # 可选的Rerank
            if self.rerank_service and aggregated_results:
                logger.info(f"[RERANK] Calling rerank service with {len(aggregated_results)} results")
                reranked_results = await self.rerank_service.rerank(sub_q, aggregated_results)
                logger.info(f"[RERANK] Returned {len(reranked_results)} results")
            else:
                reranked_results = aggregated_results

            # 证据预算控制
            trimmed_results, was_trimmed = await self.budget_controller.control_async(reranked_results, sub_q)

            if was_trimmed:
                retrieval_notes.append(f"子问题「{sub_q}」的证据超过预算，已进行裁剪")

            # 构建引用
            references = []
            for i, r in enumerate(trimmed_results[:5]):
                # 兼容 dict 和 RetrievalResult
                content = r.content if hasattr(r, 'content') else (r.get("content", "") if isinstance(r, dict) else "")
                meta = r.metadata if hasattr(r, 'metadata') else (r.get("metadata", {}) if isinstance(r, dict) else {})
                references.append({
                    "reference_id": f"ref_{idx}_{i}",
                    "document_name": meta.get("document_name", ""),
                    "section_path": meta.get("section_path", ""),
                    "channel": r.channel if hasattr(r, 'channel') else r.get("channel", "hybrid"),
                    "content_preview": content[:200]
                })

            sub_question_evidence = SubQuestionEvidence(
                sub_question_index=idx,
                sub_question=sub_q,
                references=references,
                documents=list(set([r.document_id if hasattr(r, 'document_id') else r.get("document_id", "") for r in trimmed_results])),
                fused_candidate_count=len(fused_results),
                parent_candidate_count=len(aggregated_results),
                reranked_candidate_count=len(reranked_results) if self.rerank_service else 0,
                channel_traces=[
                    {"channel_name": "vector", "recalled_count": len(vector_results), "accepted_count": len([r for r in fused_results if r.channel == "vector"])},
                    {"channel_name": "keyword", "recalled_count": len(keyword_results), "accepted_count": len([r for r in fused_results if r.channel == "keyword"])}
                ]
            )
            sub_question_evidence_list.append(sub_question_evidence)
            all_references.extend(references)

        retrieval_notes.append(f"共检索到 {len(all_references)} 条相关引用")

        return RagRetrievalContext(
            retrieval_question=retrieval_question,
            sub_question_evidence_list=sub_question_evidence_list,
            used_channels=list(used_channels),
            retrieval_notes=retrieval_notes,
            all_references=all_references
        )