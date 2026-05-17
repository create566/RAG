"""
文档服务
"""
from typing import List, Dict, Any, Optional
from app.models.document import Document, DocumentChunk, ChunkStrategy


class DocumentService:
    """文档服务 - 提供文档管理和检索能力"""

    def __init__(self, vector_store=None, keyword_store=None, config: Dict = None):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.config = config or {}

    async def list_retrievable_documents(self) -> List[Dict[str, Any]]:
        """获取可检索的文档列表（按 document_name 去重）"""
        docs_by_uuid = {}
        if self.vector_store:
            try:
                result = self.vector_store.get(limit=1000, include=["metadatas"])
                if result is None:
                    return []
                metadatas = result.get("metadatas")
                if not metadatas:
                    return []
                for metadata in metadatas:
                    doc_id = metadata.get("document_id")
                    if doc_id and doc_id not in docs_by_uuid:
                        docs_by_uuid[doc_id] = {
                            "id": doc_id,
                            "document_name": metadata.get("document_name", f"文档{doc_id}"),
                            "knowledge_scope_code": metadata.get("knowledge_scope_code", ""),
                            "knowledge_scope_name": metadata.get("knowledge_scope_name", ""),
                            "business_category": metadata.get("business_category", ""),
                            "document_tags": metadata.get("document_tags", ""),
                            "last_index_task_id": metadata.get("task_id")
                        }
            except:
                pass

        # 按 document_name 去重，保留最后出现的（最新的）
        seen_names = {}
        for doc in docs_by_uuid.values():
            name = doc["document_name"]
            seen_names[name] = doc  # 覆盖旧值，保留最新的

        return list(seen_names.values())

    async def get_document(self, document_id: int) -> Optional[Document]:
        """获取文档详情"""
        return None

    async def upload_document(self, file_path: str, metadata: Dict[str, Any]) -> Document:
        """上传文档"""
        doc = Document(
            document_name=metadata.get("name", "未命名文档"),
            knowledge_scope_code=metadata.get("scope_code"),
            knowledge_scope_name=metadata.get("scope_name"),
            business_category=metadata.get("category"),
            document_tags=metadata.get("tags"),
            file_path=file_path,
            status="pending"
        )
        return doc

    async def index_document(self, document_id: int, chunks: List[DocumentChunk]):
        """索引文档块"""
        for chunk in chunks:
            if self.vector_store:
                await self.vector_store.add(
                    documents=[chunk.content],
                    embeddings=[chunk.vector_embedding] if chunk.vector_embedding else None,
                    metadatas=[{
                        "document_id": str(chunk.document_id),
                        "chunk_id": chunk.chunk_id,
                        "parent_id": chunk.parent_id,
                        "document_name": "",
                        "section_path": chunk.metadata.get("section_path", "")
                    }],
                    ids=[chunk.chunk_id]
                )

            if self.keyword_store:
                await self.keyword_store.index(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata
                )