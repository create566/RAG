"""
文档业务服务 — 从 api/document.py 提取
"""
import os
import uuid
import re
from pathlib import Path

from app.config import get_settings
from app.utils.env import resolve_env
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentService:
    """文档上传服务 - 同步处理版本"""

    def __init__(self):
        settings = get_settings()

        from app.documents.processor import DocumentProcessor, TextSplitter
        from app.core.chroma_client import create_vector_store
        from app.core.llm_service import create_llm_service

        self.processor = DocumentProcessor()
        self.llm_service = create_llm_service(
            provider=settings.llm.provider,
            config={
                "api_key": settings.llm.api_key,
                "model": settings.llm.model,
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

        vector_cfg = {
            "persist_directory": settings.chroma.persist_directory,
            "collection_name": settings.chroma.collection_name,
        }
        self.vector_store = create_vector_store(vector_cfg)

        # ES 客户端
        es_hosts = [resolve_env(h) if isinstance(h, str) else h for h in settings.elasticsearch.hosts]
        from elasticsearch import Elasticsearch
        self.es_client = Elasticsearch(hosts=es_hosts) if es_hosts and es_hosts[0] else None
        self.es_index = settings.elasticsearch.index

        # Neo4j 客户端
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
                    logger.warning("Neo4j health check failed, disabling graph storage")
            except Exception as e:
                logger.warning(f"Neo4j init failed: {e}")

    async def process_upload(self, file, document_name: str = None, user_id: int = None, chunk_strategy: str = None):
        """处理文档上传（同步处理）"""
        from app.models.document import Document

        settings = get_settings()

        if chunk_strategy:
            from app.documents.processor import TextSplitter
            self.splitter = TextSplitter(
                strategy=chunk_strategy,
                config={
                    "parent_chunk_size": settings.retrieval.parent_chunk_size,
                    "child_chunk_size": settings.retrieval.child_chunk_size,
                    "max_child_chunks": settings.document.max_child_chunks,
                },
                llm_service=self.llm_service,
            )
            logger.info(f"使用切块策略: {chunk_strategy}")

        doc_id = str(uuid.uuid4())
        original_filename = file.filename
        file_name = document_name or original_filename

        content = await file.read()

        local_path = f"./data/uploads/{doc_id}_{original_filename}"
        os.makedirs("./data/uploads", exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(content)

        try:
            text = await self.processor.parse_document(local_path)
            if text.startswith("解析失败"):
                return Document(
                    id=int(doc_id[:8], 16) if doc_id else None,
                    document_name=file_name,
                    file_path=local_path,
                    file_size=len(content),
                    status="parse_failed",
                )

            chunks = await self.splitter.split(text)

            chunk_ids = []
            embeddings = []
            metadatas = []
            valid_chunks = []
            embedding_ok = True

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                valid_chunks.append(chunk)

                # 提取章节标题作为 section_path
                section_path = _extract_section_title(chunk, i)

                embedding = await self.llm_service.embed(chunk)
                if not embedding or len(embedding) == 0:
                    if embedding_ok:
                        logger.warning(f"Embedding 返回空，chunk={i}（仅关键词检索可用）")
                    embedding_ok = False
                    embedding = [0.0] * 1536
                elif embedding_ok and i == 0:
                    logger.info(f"Embedding 成功: dim={len(embedding)}")
                embeddings.append(embedding)
                metadatas.append({
                    "document_id": doc_id,
                    "document_name": file_name,
                    "chunk_id": chunk_id,
                    "section_path": section_path,
                    "user_id": user_id if user_id else 0,
                })

            if chunk_ids:
                self.vector_store.add(
                    documents=valid_chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=chunk_ids,
                )

            self._index_to_es(doc_id, file_name, valid_chunks, metadatas)
            self._index_to_neo4j(doc_id, file_name, valid_chunks)

            status = "indexed" if embedding_ok else "keyword_only"
            if not embedding_ok:
                logger.warning("Embedding 不可用，文档仅支持关键词检索")

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            import traceback
            traceback.print_exc()
            status = "error"

        return Document(
            id=int(doc_id[:8], 16) if doc_id else None,
            user_id=user_id if user_id else 0,
            document_name=file_name,
            file_path=local_path,
            file_size=len(content),
            status=status,
        )

    async def queue_upload(self, file, document_name: str = None, user_id: int = None, chunk_strategy: str = None):
        """将文档放入 Redis Stream 异步队列"""
        from app.models.document import Document

        doc_id = str(uuid.uuid4())
        original_filename = file.filename
        file_name = document_name or original_filename

        content = await file.read()
        local_path = f"./data/uploads/{doc_id}_{original_filename}"
        os.makedirs("./data/uploads", exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(content)

        # 发送消息到 Redis Stream
        from app.core.redis_client import create_redis_cache
        from app.config import get_settings
        from app.utils.env import resolve_env
        settings = get_settings()
        redis_cfg = {}
        if hasattr(settings, 'redis'):
            redis_cfg = {
                "host": resolve_env(settings.redis.host) if hasattr(settings.redis, 'host') else "localhost",
                "port": settings.redis.port if hasattr(settings.redis, 'port') else 6379,
                "db": settings.redis.db if hasattr(settings.redis, 'db') else 0,
            }
        redis_cache = create_redis_cache(redis_cfg)
        await redis_cache.xadd("document_processing", {
            "doc_id": doc_id,
            "file_path": local_path,
            "file_name": file_name,
            "user_id": str(user_id) if user_id else "0",
            "chunk_strategy": chunk_strategy or "",
        })
        await redis_cache.close()

        return Document(
            id=int(doc_id[:8], 16) if doc_id else None,
            user_id=user_id if user_id else 0,
            document_name=file_name,
            file_path=local_path,
            file_size=len(content),
            status="queued",
        )

    async def list_documents(self, user_id: int = None):
        """列出已上传的文档"""
        from app.models.document import Document

        try:
            result = self.vector_store.peek(limit=1000)
            docs = {}
            metadatas = result.get("metadatas", []) if result else []
            for metadata in metadatas:
                doc_user_id = metadata.get("user_id")
                if doc_user_id is not None and user_id is not None and user_id > 0:
                    if int(doc_user_id) != int(user_id):
                        continue
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = Document(
                        id=int(doc_id[:8], 16) if doc_id else None,
                        user_id=int(doc_user_id) if doc_user_id is not None else 0,
                        document_name=metadata.get("document_name", "未命名"),
                        file_path=metadata.get("file_path", ""),
                        file_size=0,
                        status="indexed",
                    )
            return list(docs.values())
        except Exception as e:
            logger.error(f"List documents error: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """删除文档 — 级联删除向量、ES、Neo4j 和本地文件"""
        try:
            target_uuid = None
            is_uuid = len(document_id) == 36 and "-" in document_id

            if is_uuid:
                target_uuid = document_id
            else:
                try:
                    target_int = int(document_id)
                except ValueError:
                    target_int = None

                result = self.vector_store.peek(limit=1000)
                metadatas = result.get("metadatas", []) if result else []
                for meta in metadatas:
                    doc_id = meta.get("document_id", "")
                    if isinstance(doc_id, str) and len(doc_id) == 36:
                        try:
                            if int(doc_id[:8], 16) == target_int:
                                target_uuid = doc_id
                                break
                        except ValueError:
                            pass

            if not target_uuid:
                logger.warning(f"Cannot resolve document_id={document_id} to UUID")
                return False

            logger.info(f"DELETE resolved: {document_id} -> UUID={target_uuid}")

            # Chroma
            self.vector_store.delete(where={"document_id": target_uuid})

            # Elasticsearch
            if self.es_client:
                try:
                    self.es_client.delete_by_query(
                        index=self.es_index,
                        body={"query": {"term": {"document_id": target_uuid}}},
                    )
                except Exception as e:
                    logger.warning(f"ES delete error: {e}")

            # Neo4j
            if self.neo4j_client:
                try:
                    doc_int_id = int(target_uuid[:8], 16)
                    self.neo4j_client.delete_document(doc_int_id)
                except Exception as e:
                    logger.warning(f"Neo4j delete error: {e}")

            # 本地文件
            uploads_dir = "./data/uploads"
            if os.path.exists(uploads_dir):
                uuid_prefix = target_uuid[:8]
                for filename in os.listdir(uploads_dir):
                    if filename.startswith(uuid_prefix):
                        filepath = os.path.join(uploads_dir, filename)
                        if os.path.isfile(filepath):
                            os.remove(filepath)
                            logger.info(f"Local file deleted: {filepath}")
                            break

            return True
        except Exception as e:
            logger.error(f"Delete document error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _index_to_es(self, doc_id: str, document_name: str, chunks: list, metadatas: list):
        """将文档块写入 ES 索引"""
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
                self.es_client.index(
                    index=self.es_index,
                    id=meta.get("chunk_id", ""),
                    body=doc,
                )
            logger.info(f"ES indexed {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"ES index error: {e}")

    def _index_to_neo4j(self, doc_id: str, document_name: str, chunks: list):
        """将文档块写入 Neo4j 图谱"""
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

            logger.info(f"Neo4j indexed {len(chapters)} chapters")
        except Exception as e:
            logger.error(f"Neo4j index error: {e}")

    def _extract_chapters(self, chunks: list) -> list:
        """从 chunks 中提取章节信息"""
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
    """从chunk内容中提取章节标题作为section_path"""
    lines = chunk.strip().split("\n")
    for line in lines[:5]:  # 只检查前5行
        line = line.strip()
        # 匹配 ## 标题
        if line.startswith("## "):
            return line.replace("## ", "").strip()[:50]
        # 匹配 第X章 标题
        if re.match(r"^第[一二三四五六七八九十\d]+章", line):
            return line.strip()[:50]
        # 匹配 一、标题 或 1. 标题
        if re.match(r"^[一二三四五六七八九十\d]+[、、.]", line):
            return line.strip()[:50]
    return f"第{index + 1}节"


_document_service = None


def get_document_service():
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service
