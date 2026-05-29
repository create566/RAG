"""
Chroma 向量数据库客户端
用于文档的向量存储和检索
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """Chroma 向量存储客户端"""

    def __init__(self, user_id: int = None, persist_directory: str = "./data/chroma", collection_name: str = None):
        self.user_id = user_id
        self.persist_directory = persist_directory
        # 用户隔离：每个用户独立的 collection
        if collection_name:
            self.collection_name = collection_name
        elif user_id:
            self.collection_name = f"user_{user_id}_docs"
        else:
            self.collection_name = "super_agent_docs"
        self._client = None
        self._collection = None

    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
        return self._client

    @property
    def collection(self):
        """获取集合"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Super Agent Document Collection"}
                )
        return self._collection

    def add(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict], ids: List[str]):
        """添加文档到集合"""
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_embeddings: List[List[float]], n_results: int = 10, where: Dict = None) -> Dict:
        """查询相似文档"""
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

    def get(self, ids: List[str] = None, limit: int = 100, include: List[str] = None, where: Dict = None) -> Dict:
        """获取文档"""
        return self.collection.get(
            ids=ids,
            limit=limit,
            include=include or ["documents", "metadatas", "embeddings"],
            where=where
        )

    def delete(self, ids: List[str] = None, where: Dict = None):
        """删除文档"""
        self.collection.delete(ids=ids, where=where)

    def peek(self, limit: int = 10) -> Dict:
        """预览集合"""
        return self.collection.peek(limit=limit)

    def count(self) -> int:
        """文档数量"""
        return self.collection.count()


def create_vector_store(user_id: int = None, config: Dict = None) -> ChromaVectorStore:
    """Factory method to create vector store (user isolated)"""
    config = config or {}
    return ChromaVectorStore(
        user_id=user_id,
        persist_directory=config.get("persist_directory", "./data/chroma"),
        collection_name=config.get("collection_name")
    )


class ConversationMemoryStore:
    """对话记忆向量存储"""

    COLLECTION_NAME = "conversation_memory"

    def __init__(self, persist_directory: str = "./data/chroma"):
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(name=self.COLLECTION_NAME)
            except Exception:
                self._collection = self.client.create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"description": "Conversation Memory Vector Store"}
                )
        return self._collection

    def add(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict], ids: List[str]):
        """添加对话记忆"""
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_embeddings: List[List[float]], n_results: int = 5, where: Dict = None) -> Dict:
        """检索相似记忆"""
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

    def delete_by_conversation(self, conversation_id: str):
        """删除某会话的所有记忆"""
        self.collection.delete(where={"conversation_id": conversation_id})

    def count(self) -> int:
        return self.collection.count()


def create_conversation_memory_store(persist_directory: str = "./data/chroma") -> ConversationMemoryStore:
    """工厂方法创建对话记忆存储"""
    return ConversationMemoryStore(persist_directory=persist_directory)