"""
文档管理API接口 - 单体项目版本
上传后直接同步处理（解析、切块、向量化）
"""
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import uuid
import re
from pathlib import Path

from app.models.document import Document, DocumentChunk
from app.documents.processor import DocumentProcessor, TextSplitter
from app.core.chroma_client import create_vector_store
from app.core.llm_service import create_llm_service

router = APIRouter(prefix="/api/document", tags=["document"])


class DocumentUploadService:
    """文档上传服务 - 同步处理版本"""

    def __init__(self):
        self.config = self._load_config()
        self.processor = DocumentProcessor()
        # LLM 服务（先生成，供 TextSplitter 的 llm 策略使用）
        self.llm_service = create_llm_service(
            provider=self.config.get("llm", {}).get("provider", "dashscope"),
            config=self.config.get("llm", {})
        )
        # 取文档配置中的 chunk_strategies 作为默认策略
        doc_config = self.config.get("document", {})
        # 优先使用 default_strategy（可以是逗号分隔的组合，如 "structural,recursive,semantic"）
        default_strategy = doc_config.get("default_strategy", "structural")
        self.splitter = TextSplitter(
            strategy=default_strategy,
            config=doc_config,
            llm_service=self.llm_service
        )
        # 向量存储配置
        vector_config = self.config.get("vector_store", {})
        if vector_config.get("chroma"):
            vector_config = {**vector_config, **vector_config.get("chroma", {})}
        self.vector_store = create_vector_store(vector_config)

        # ES 客户端（关键词检索用）
        keyword_config = self.config.get("keyword_search", {}).get("elasticsearch", {})
        es_hosts_raw = keyword_config.get("hosts", ["http://localhost:9200"])

        # 解析环境变量 ${VAR:-default} 格式
        def resolve_env(value):
            if isinstance(value, str):
                if value.startswith("${") and ":-" in value and value.endswith("}"):
                    inner = value[2:-1]
                    var_name, default = inner.split(":-", 1)
                    return os.environ.get(var_name, default)
                elif value.startswith("${") and value.endswith("}"):
                    return os.environ.get(value[2:-1], "")
            return value

        es_hosts = [resolve_env(h) if isinstance(h, str) else h for h in es_hosts_raw]

        if es_hosts and isinstance(es_hosts, list) and es_hosts[0] and es_hosts[0] != "${ELASTICSEARCH_HOSTS:-http://localhost:9200}":
            from elasticsearch import Elasticsearch
            self.es_client = Elasticsearch(hosts=es_hosts)
            self.es_index = keyword_config.get("index", "super_agent_keywords")
        else:
            self.es_client = None
            self.es_index = "super_agent_keywords"

        # Neo4j 客户端（图谱构建用）
        graph_config = self.config.get("graph_db", {})
        if graph_config.get("neo4j", {}).get("uri"):
            try:
                from app.core.neo4j_client import Neo4jClient
                neo4j_config = graph_config["neo4j"]
                self.neo4j_client = Neo4jClient(
                    uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                    username=neo4j_config.get("username", "neo4j"),
                    password=neo4j_config.get("password", "")
                )
                if not self.neo4j_client.health_check():
                    self.neo4j_client = None
                    print("[NEO4J] Health check failed, disabling graph storage")
            except Exception as e:
                self.neo4j_client = None
                print(f"[NEO4J] Init failed: {e}")
        else:
            self.neo4j_client = None

    def _load_config(self) -> dict:
        """加载配置"""
        try:
            import yaml
            from pathlib import Path
            from dotenv import load_dotenv

            # 加载 .env 文件
            load_dotenv(Path(__file__).parent.parent.parent / ".env")

            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}

                # 解析环境变量
                def resolve_env(value):
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_key = value[2:-1]
                        return os.environ.get(env_key, "")
                    return value

                # 解析 llm 配置
                llm = config.get("llm", {})
                dashscope = llm.get("dashscope", {})
                dashscope["api_key"] = resolve_env(dashscope.get("api_key", ""))
                llm["dashscope"] = dashscope

                # 解析 embedding 配置
                embedding = config.get("embedding", {})
                dashscope_emb = embedding.get("dashscope", {})
                dashscope_emb["api_key"] = resolve_env(dashscope_emb.get("api_key", ""))
                embedding["dashscope"] = dashscope_emb

                return config
        except:
            pass
        return {}

    def _index_to_es(self, doc_id: str, document_name: str, chunks: list, metadatas: list):
        """将文档块写入 ES 索引（用于 BM25 关键词检索）"""
        if not self.es_client:
            print("[ES] ES 客户端未初始化，跳过索引")
            return

        try:
            # 确保索引存在
            if not self.es_client.indices.exists(index=self.es_index):
                self.es_client.indices.create(
                    index=self.es_index,
                    body={
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "default": {
                                        "type": "standard"
                                    }
                                }
                            }
                        },
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
                                "user_id": {"type": "long"}
                            }
                        }
                    }
                )
                print(f"[ES] Created index: {self.es_index}")

            # 批量写入文档
            for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
                doc = {
                    "content": chunk,
                    "document_id": meta.get("document_id", ""),
                    "document_name": meta.get("document_name", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "parent_id": meta.get("parent_id", ""),
                    "section_path": meta.get("section_path", f"chunk_{i}"),
                    "user_id": meta.get("user_id", 0)
                }
                self.es_client.index(
                    index=self.es_index,
                    id=meta.get("chunk_id", ""),
                    body=doc
                )

            print(f"[ES] Indexed {len(chunks)} chunks to {self.es_index}")

        except Exception as e:
            print(f"[ES] Index error: {e}")
            import traceback
            traceback.print_exc()

    def _index_to_neo4j(self, doc_id: str, document_name: str, chunks: list):
        """将文档块写入 Neo4j 图谱"""
        if not self.neo4j_client:
            print("[NEO4J] Client not initialized, skipping")
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

            print(f"[NEO4J] Indexed {len(chapters)} chapters for doc {doc_id}")
        except Exception as e:
            print(f"[NEO4J] Index error: {e}")
            import traceback
            traceback.print_exc()

    def _extract_chapters(self, chunks: list) -> list:
        """从 chunks 中提取章节信息 - 按 '第X章' 或 '##' 标题切分"""
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

    async def process_upload(self, file: UploadFile, document_name: str = None, user_id: int = None, chunk_strategy: str = None) -> Document:
        """
        处理文档上传（同步处理）：
        1. 保存到本地
        2. 解析文档
        3. 切块（可选指定策略）
        4. 向量化
        5. 存储到 Chroma
        """
        # 如果指定了策略，动态切换 splitter
        if chunk_strategy:
            doc_config = self.config.get("document", {})
            self.splitter = TextSplitter(
                strategy=chunk_strategy,
                config=doc_config,
                llm_service=self.llm_service
            )
            print(f"[UPLOAD] 使用策略: {chunk_strategy}")
        # 生成唯一ID
        doc_id = str(uuid.uuid4())
        original_filename = file.filename
        file_name = document_name or original_filename

        # 读取文件内容
        content = await file.read()

        # 保存到本地
        local_path = f"./data/uploads/{doc_id}_{original_filename}"
        os.makedirs("./data/uploads", exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(content)

        try:
            # 2. 解析文档
            text = await self.processor.parse_document(local_path)
            if text.startswith("解析失败"):
                return Document(
                    id=int(doc_id[:8], 16) if doc_id else None,
                    document_name=file_name,
                    file_path=local_path,
                    file_size=len(content),
                    status="parse_failed"
                )

            # 3. 切分文本
            chunks = await self.splitter.split(text)

            # 4. 生成向量并存储
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

                # 获取嵌入向量
                embedding = await self.llm_service.embed(chunk)
                if not embedding or len(embedding) == 0:
                    embedding_ok = False
                    # 使用零向量作为占位（不影响关键词检索）
                    embedding = [0.0] * 1536

                embeddings.append(embedding)

                # 元数据
                metadatas.append({
                    "document_id": doc_id,
                    "document_name": file_name,
                    "chunk_id": chunk_id,
                    "section_path": f"chunk_{i}",
                    "user_id": user_id if user_id else 0
                })

            # 5. 存储到向量数据库（始终存储以追踪文档）
            if chunk_ids:
                self.vector_store.add(
                    documents=valid_chunks,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=chunk_ids
                )

            # 6. 同时写入 ES 索引（BM25 关键词检索）
            self._index_to_es(doc_id, file_name, valid_chunks, metadatas)

            # 7. 写入 Neo4j 图谱
            self._index_to_neo4j(doc_id, file_name, valid_chunks)

            status = "indexed" if embedding_ok else "keyword_only"
            if not embedding_ok:
                print("[WARN] Embedding不可用，文档仅支持关键词检索")

        except Exception as e:
            print(f"Document processing error: {e}")
            import traceback
            traceback.print_exc()
            status = "error"

        # 返回文档信息
        return Document(
            id=int(doc_id[:8], 16) if doc_id else None,
            user_id=user_id if user_id else 0,
            document_name=file_name,
            file_path=local_path,
            file_size=len(content),
            status=status
        )

    async def list_documents(self, user_id: int = None) -> List[Document]:
        """列出已上传的文档，可按用户筛选"""
        try:
            count = self.vector_store.count()
            print(f"[DOC LIST] Chroma count={count}")
            result = self.vector_store.peek(limit=1000)
            docs = {}
            metadatas = result.get("metadatas", []) if result else []
            print(f"[DOC LIST] metadatas count={len(metadatas)}, user_id filter={user_id}")
            if metadatas:
                print(f"[DOC LIST] first meta sample: {metadatas[0]}")
            for metadata in metadatas:
                doc_user_id = metadata.get("user_id")
                # 兼容: Chroma 可能把 int 存成 float (2.0)
                # user_id 为 0/None 时不过滤（显示全部文档）
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
                        status="indexed"
                    )
            return list(docs.values())
        except Exception as e:
            print(f"List documents error: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """删除文档 - 级联删除向量、ES、Neo4j和本地文件"""
        try:
            # 1. 解析 document_id：可能是 UUID 字符串或 int 表示
            target_uuid = None
            is_uuid = len(document_id) == 36 and "-" in document_id

            if is_uuid:
                target_uuid = document_id
            else:
                # 前端传的是 int(doc_id[:8], 16) 的整数形式，从 Chroma 查找对应 UUID
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
                print(f"[DELETE] Cannot resolve document_id={document_id} to UUID")
                return False

            print(f"[DELETE] Resolved: {document_id} -> UUID={target_uuid}")

            # 2. 获取文档信息（用于删除本地文件）
            result = self.vector_store.peek(limit=1000)
            metadatas = result.get("metadatas", []) if result else []
            doc_to_delete = None
            for meta in metadatas:
                if meta.get("document_id") == target_uuid:
                    doc_to_delete = meta
                    break

            # 3. 删除 Chroma 向量数据库（用 UUID）
            self.vector_store.delete(where={"document_id": target_uuid})
            print(f"[DELETE] Chroma: deleted document_id={target_uuid}")

            # 4. 删除 Elasticsearch BM25 数据（用 UUID）
            if self.es_client:
                try:
                    self.es_client.delete_by_query(
                        index=self.es_index,
                        body={"query": {"term": {"document_id": target_uuid}}}
                    )
                    print(f"[DELETE] ES: deleted document_id={target_uuid}")
                except Exception as e:
                    print(f"[DELETE] ES delete error: {e}")

            # 5. 删除 Neo4j 图数据（用 int 形式）
            if self.neo4j_client:
                try:
                    doc_int_id = int(target_uuid[:8], 16)
                    self.neo4j_client.delete_document(doc_int_id)
                    print(f"[DELETE] Neo4j: deleted document_id={doc_int_id}")
                except Exception as e:
                    print(f"[DELETE] Neo4j delete error: {e}")

            # 6. 删除本地文件
            local_file_deleted = False
            if doc_to_delete and doc_to_delete.get("file_path"):
                local_path = doc_to_delete.get("file_path")
                if os.path.exists(local_path):
                    os.remove(local_path)
                    print(f"[DELETE] Local file: deleted {local_path}")
                    local_file_deleted = True

            if not local_file_deleted:
                uploads_dir = "./data/uploads"
                if os.path.exists(uploads_dir):
                    uuid_prefix = target_uuid[:8]
                    for filename in os.listdir(uploads_dir):
                        if filename.startswith(uuid_prefix):
                            filepath = os.path.join(uploads_dir, filename)
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                                print(f"[DELETE] Local file: deleted {filepath}")
                                local_file_deleted = True
                                break

            return True
        except Exception as e:
            print(f"Delete document error: {e}")
            import traceback
            traceback.print_exc()
            return False


# 全局服务实例
upload_service = DocumentUploadService()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = None,
    user_id: Optional[int] = None,
    chunk_strategy: Optional[str] = None
):
    """上传文档（同步处理），可选指定切块策略"""
    print(f"[UPLOAD API] received: file={file.filename}, user_id={user_id}, chunk_strategy={chunk_strategy}")
    try:
        # 检查文件类型
        allowed_exts = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".xlsx", ".xls"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

        # 处理上传（同步处理），传入切块策略
        doc = await upload_service.process_upload(file, document_name, user_id, chunk_strategy)
        return {
            "success": True,
            "document": {
                "id": doc.id,
                "document_name": doc.document_name,
                "file_size": doc.file_size,
                "status": doc.status,
                "chunk_strategy": chunk_strategy or upload_service.config.get("document", {}).get("default_strategy", "structural,recursive")
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(user_id: Optional[int] = None):
    """列出所有已上传的文档（可按用户筛选）"""
    try:
        docs = await upload_service.list_documents(user_id)
        return {
            "success": True,
            "documents": [
                {
                    "id": d.id,
                    "user_id": d.user_id,
                    "document_name": d.document_name,
                    "status": d.status
                }
                for d in docs
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    success = await upload_service.delete_document(document_id)
    return {"success": success}