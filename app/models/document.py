"""
文档相关的数据模型
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ChunkStrategy(str, Enum):
    """切块策略"""
    STRUCTURAL = "structural"     # 基于文档结构
    RECURSIVE = "recursive"       # 递归分块
    SEMANTIC = "semantic"         # 语义分块
    LLM = "llm"                   # LLM智能切块


class Document(BaseModel):
    """文档"""
    id: Optional[int] = None
    user_id: Optional[int] = 0  # 用户ID，关联上传者（兼容旧数据无此字段）
    document_name: str
    knowledge_scope_code: Optional[str] = None
    knowledge_scope_name: Optional[str] = None
    business_category: Optional[str] = None
    document_tags: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_index_task_id: Optional[int] = None


class DocumentChunk(BaseModel):
    """文档块"""
    id: Optional[int] = None
    document_id: int
    parent_id: Optional[str] = None
    chunk_id: str
    content: str
    chunk_index: int
    vector_embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeDomain(BaseModel):
    """知识域"""
    code: str
    name: str
    description: Optional[str] = None
    document_count: int = 0


class DocumentRouteCandidate(BaseModel):
    """文档路由候选"""
    document_id: str
    document_name: str
    last_index_task_id: str
    knowledge_scope_code: str
    knowledge_scope_name: str
    business_category: str
    document_tags: str
    score: float
    reason: str


class KnowledgeRouteDecision(BaseModel):
    """知识路由决策"""
    confidence: float
    route_status: str
    documents: List[DocumentRouteCandidate]