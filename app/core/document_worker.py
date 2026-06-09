"""
文档处理 Worker — 消费 Redis Stream 异步队列
"""
import asyncio
import uuid
import re
import os
from typing import Dict, List

from app.config import get_settings
from app.utils.env import resolve_env
from app.core.logging import get_logger
from app.core.redis_client import create_redis_cache
from app.core.chroma_client import create_vector_store
from app.core.llm_service import create_llm_service
from app.documents.processor import DocumentProcessor, TextSplitter

logger = get_logger(__name__)

STREAM_NAME = "document_processing"
CONSUMER_GROUP = "document-workers"


class DocumentWorker:
    """文档处理 Worker：从 Redis Stream 消费消息并处理"""

    def __init__(self):
        settings = get_settings()
        redis_cfg = {}
        if hasattr(settings, 'redis'):
            redis_cfg = {
                "host": resolve_env(settings.redis.host) if hasattr(settings.redis, 'host') else "localhost",
                "port": settings.redis.port if hasattr(settings.redis, 'port') else 6379,
                "db": settings.redis.db if hasattr(settings.redis, 'db') else 0,
            }
        self.redis_cache = create_redis_cache(redis_cfg)
        self.processor = DocumentProcessor()
        self.llm_service = create_llm_service(
            provider=settings.llm.provider,
            config={
                "api_key": settings.llm.api_key,
                "model": settings.llm.model,
                # 独立的 embedding 配置（embedding provider 可能与 LLM provider 不同）
                "embed_model": settings.embedding.model,
                "embed_base_url": settings.embedding.base_url,
                "embed_api_key": settings.embedding.api_key,
            },
        )
        self.splitter = TextSplitter(
            strategy=settings.document.default_strategy,
            config={
                "parent_chunk_size": settings.retrieval.parent_chunk_size,
                "child_chunk_size": settings.retrieval.child_chunk_size,
                "max_child_chunks": settings.document.max_child_chunks,
            },
            llm_service=self.llm_service,
        )
        self.vector_store = create_vector_store(config={
            "persist_directory": settings.chroma.persist_directory,
            "collection_name": settings.chroma.collection_name,
        })
        self._running = False
        self._worker_name = f"worker-{uuid.uuid4().hex[:8]}"

        # ES 客户端
        es_hosts = [resolve_env(h) if isinstance(h, str) else h for h in settings.elasticsearch.hosts]
        from elasticsearch import Elasticsearch
        self.es_client = Elasticsearch(hosts=es_hosts) if es_hosts and es_hosts[0] else None
        self.es_index = settings.elasticsearch.index

        # Neo4j
        self.neo4j_client = None
        if settings.neo4j.uri:
            try:
                from app.core.neo4j_client import Neo4jClient
                self.neo4j_client = Neo4jClient(
                    uri=settings.neo4j.uri,
                    username=settings.neo4j.username,
                    password=settings.neo4j.password,
                )
                if not self.neo4j_client.health_check():
                    self.neo4j_client = None
            except Exception as e:
                logger.warning(f"Neo4j init failed: {e}")

    async def start(self):
        """启动 Worker"""
        await self.redis_cache.xgroup_create(STREAM_NAME, CONSUMER_GROUP)
        self._running = True
        logger.info(f"[Worker] {self._worker_name} 已启动，等待消息...")

        while self._running:
            messages = await self.redis_cache.xread_group(
                CONSUMER_GROUP, self._worker_name, [STREAM_NAME], count=5, block=5000
            )
            for msg in messages:
                await self._handle_message(msg)

    async def stop(self):
        """停止 Worker"""
        self._running = False
        await self.redis_cache.close()

    async def _handle_message(self, msg: Dict):
        """处理单条消息"""
        msg_id = msg.get("id", "?")
        data = msg.get("data", {})
        doc_id = data.get("doc_id", "?")
        logger.info(f"[Worker] 处理文档: doc_id={doc_id}, msg_id={msg_id}")

        try:
            file_path = data.get("file_path")
            file_name = data.get("file_name", "未命名")
            user_id = int(data.get("user_id", 0))
            chunk_strategy = data.get("chunk_strategy", None)

            if not file_path or not os.path.exists(file_path):
                logger.error(f"[Worker] 文件不存在: {file_path}")
                await self.redis_cache.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                return

            # 解析文档
            text = await self.processor.parse_document(file_path)
            if text.startswith("解析失败"):
                logger.error(f"[Worker] 解析失败: {text}")
                await self.redis_cache.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                return

            # 切块
            if chunk_strategy:
                splitter = TextSplitter(
                    strategy=chunk_strategy,
                    config={
                        "parent_chunk_size": get_settings().retrieval.parent_chunk_size,
                        "child_chunk_size": get_settings().retrieval.child_chunk_size,
                        "max_child_chunks": get_settings().document.max_child_chunks,
                    },
                    llm_service=self.llm_service,
                )
            else:
                splitter = self.splitter

            chunks = await splitter.split(text)

            # 向量化并存储
            chunk_ids, embeddings, metadatas, valid_chunks = [], [], [], []
            embedding_ok = True

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                valid_chunks.append(chunk)
                section_path = _extract_section_title(chunk, i)
                logger.info(f"[Worker] 开始向量化 chunk={i}, length={len(chunk)}")
                embedding = await self.llm_service.embed(chunk)
                if not embedding or len(embedding) == 0:
                    logger.warning(f"[Worker] Embedding 返回空，chunk={i}")
                    embedding_ok = False
                    embedding = [0.0] * get_settings().embedding.dimension
                else:
                    logger.info(f"[Worker] Embedding 成功 chunk={i}, dim={len(embedding)}")
                embeddings.append(embedding)
                metadatas.append({
                    "document_id": doc_id,
                    "document_name": file_name,
                    "chunk_id": chunk_id,
                    "section_path": section_path,
                    "user_id": user_id,
                })

            if chunk_ids:
                logger.info(f"[Worker] 开始存储向量: chunks={len(chunk_ids)}, embedding_ok={embedding_ok}")
                self.vector_store.add(valid_chunks, embeddings, metadatas, chunk_ids)
                logger.info(f"[Worker] 向量存储完成: collection={self.vector_store.collection.count()}")

            # 写入 ES 索引
            self._index_to_es(doc_id, file_name, valid_chunks, metadatas)

            # 写入 Neo4j 图谱
            self._index_to_neo4j(doc_id, file_name, valid_chunks)

            status = "indexed" if embedding_ok else "keyword_only"
            logger.info(f"[Worker] 文档处理完成: doc_id={doc_id}, status={status}")

            # 清理临时文件
            try:
                os.remove(file_path)
            except:
                pass

        except Exception as e:
            logger.error(f"[Worker] 处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.redis_cache.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)

    def _index_to_es(self, doc_id: str, document_name: str, chunks: list, metadatas: list):
        if not self.es_client:
            return
        try:
            if not self.es_client.indices.exists(index=self.es_index):
                self.es_client.indices.create(
                    index=self.es_index,
                    body={
                        "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
                        "mappings": {
                            "properties": {
                                "content": {"type": "text"},
                                "document_id": {"type": "keyword"},
                                "document_name": {"type": "keyword"},
                                "chunk_id": {"type": "keyword"},
                                "parent_id": {"type": "keyword"},
                                "section_path": {"type": "keyword"},
                                "title": {"type": "text"},
                                "tags": {"type": "keyword"},
                                "user_id": {"type": "long"},
                            }
                        },
                    },
                )
            for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                doc = {
                    "content": chunk,
                    "document_id": meta.get("document_id", ""),
                    "document_name": meta.get("document_name", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "section_path": meta.get("section_path", f"第{i + 1}节"),
                    "user_id": meta.get("user_id", 0),
                }
                self.es_client.index(index=self.es_index, id=meta.get("chunk_id", ""), body=doc)
            logger.info(f"[Worker] ES indexed {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"[Worker] ES index error: {e}")

    def _index_to_neo4j(self, doc_id: str, document_name: str, chunks: list):
        if not self.neo4j_client:
            return
        try:
            doc_int_id = int(doc_id[:8], 16)
            self.neo4j_client.create_document_node(doc_int_id, document_name)
            chapters = self._extract_chapters(chunks)
            for chapter_title, chapter_content in chapters:
                self.neo4j_client.create_chapter_relationship(doc_int_id, chapter_title, chapter_content)
                for p_idx, para in enumerate(chapter_content.split("\n")):
                    if para.strip():
                        self.neo4j_client.create_paragraph_relationship(chapter_title, para.strip(), p_idx)
            logger.info(f"[Worker] Neo4j indexed {len(chapters)} chapters")
        except Exception as e:
            logger.error(f"[Worker] Neo4j index error: {e}")

    def _extract_chapters(self, chunks: list) -> list:
        chapters = []
        current = {"title": "前言", "content": ""}
        for chunk in chunks:
            for line in chunk.split("\n"):
                if re.match(r"^第[一二三四五六七八九十\d]+章", line) or line.startswith("## "):
                    if current["content"]:
                        chapters.append((current["title"], current["content"]))
                    current = {"title": line.strip(), "content": ""}
                else:
                    current["content"] += line + "\n"
        if current["content"]:
            chapters.append((current["title"], current["content"]))
        return chapters or [("全文", "\n".join(chunks))]


def _extract_section_title(chunk: str, index: int) -> str:
    lines = chunk.strip().split("\n")
    for line in lines[:5]:
        line = line.strip()
        if line.startswith("## "):
            return line.replace("## ", "").strip()[:50]
        if re.match(r"^第[一二三四五六七八九十\d]+章", line):
            return line.strip()[:50]
        if re.match(r"^[一二三四五六七八九十\d]+[、、.]", line):
            return line.strip()[:50]
    return f"第{index + 1}节"


_worker = None


def get_document_worker() -> DocumentWorker:
    global _worker
    if _worker is None:
        _worker = DocumentWorker()
    return _worker