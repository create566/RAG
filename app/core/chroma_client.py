"""
Chroma 向量数据库客户端
用于文档的向量存储和检索
"""
import os
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# Monkey-patch: 修复 chromadb 0.5.0 posthog 遥测兼容性问题
# patch posthog.capture 无效（Posthog.__init__ 会通过 disabled=True 覆盖），直接替换 _direct_capture
import chromadb.telemetry.product.posthog as _chroma_ph
_chroma_ph.Posthog._direct_capture = lambda self, event: None
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """Chroma 向量存储客户端 — 延迟初始化，共享底层 ChromaDB 客户端"""

    # 全局共享的 PersistentClient（同目录只需一个，避免 SQLite 锁冲突）
    _shared_clients: dict = {}

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
        self._initialized = False

    @property
    def client(self):
        """延迟初始化客户端（全局共享 PersistentClient）"""
        if self._client is None:
            key = self.persist_directory
            if key not in ChromaVectorStore._shared_clients:
                ChromaVectorStore._shared_clients[key] = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
            self._client = ChromaVectorStore._shared_clients[key]
        return self._client

    @property
    def collection(self):
        """获取集合（首次访问时真正初始化 ChromaDB）"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Super Agent Document Collection"}
                )
            if not self._initialized:
                self._initialized = True
                logger.info(f"[CHROMA] 向量数据库初始化成功 | collection={self.collection_name}, user_id={self.user_id}, persist_dir={self.persist_directory}")
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


# 向量存储实例缓存：key=(persist_dir, collection_name)，避免重复创建
_vector_store_cache: dict = {}


def create_vector_store(user_id: int = None, config: Dict = None) -> ChromaVectorStore:
    """Factory method to create vector store (user isolated) — 带缓存"""
    config = config or {}
    persist_dir = config.get("persist_directory", "./data/chroma")
    coll_name = config.get("collection_name")
    if not coll_name:
        coll_name = f"user_{user_id}_docs" if user_id else "super_agent_docs"

    cache_key = (persist_dir, coll_name)
    if cache_key not in _vector_store_cache:
        store = ChromaVectorStore(
            user_id=user_id,
            persist_directory=persist_dir,
            collection_name=coll_name,
        )
        _vector_store_cache[cache_key] = store
        logger.info(f"[CHROMA] 向量存储创建成功 | collection={store.collection_name}")
    return _vector_store_cache[cache_key]


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
            key = self.persist_directory
            if key not in ChromaVectorStore._shared_clients:
                ChromaVectorStore._shared_clients[key] = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False)
                )
            self._client = ChromaVectorStore._shared_clients[key]
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