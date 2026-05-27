"""
聊天相关的数据模型
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ChatQueryMode(str, Enum):
    """聊天模式"""
    OPEN_CHAT = "OPEN_CHAT"           # 开放式Agent
    AUTO_DOCUMENT = "AUTO_DOCUMENT"    # 自动文档路由
    DOCUMENT = "DOCUMENT"             # 指定文档问答


class ExecutionMode(str, Enum):
    """执行模式"""
    CLARIFICATION = "CLARIFICATION"   # 歧义澄清
    RETRIEVAL = "RETRIEVAL"           # RAG检索
    GRAPH_ONLY = "GRAPH_ONLY"          # 仅图查询
    GRAPH_THEN_EVIDENCE = "GRAPH_THEN_EVIDENCE"  # 图查询后证据
    REACT_AGENT = "REACT_AGENT"       # ReAct Agent


class Message(BaseModel):
    """消息"""
    role: Literal["user", "assistant", "system"] = "user"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """聊天请求"""
    user_id: int
    conversation_id: Optional[str] = None
    question: str
    chat_mode: ChatQueryMode = ChatQueryMode.AUTO_DOCUMENT
    selected_document_id: Optional[int] = None
    selected_document_name: Optional[str] = None
    selected_task_id: Optional[int] = None


class SourceReference(BaseModel):
    """来源引用"""
    reference_id: str
    document_name: str
    section_path: str
    channel: str
    score: Optional[float] = None
    content: Optional[str] = None  # 实际chunk内容，用于评估


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    sources: List[SourceReference] = []
    suggested_questions: List[str] = []
    conversation_id: str
    trace: Optional[Dict[str, Any]] = None


class ConversationSession(BaseModel):
    """会话会话"""
    id: str
    user_id: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_exchange_id: Optional[int] = None


class StageBenchmark(BaseModel):
    """阶段性能基准"""
    stage_code: str
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatDebugTrace(BaseModel):
    """聊天调试追踪"""
    retrieval_notes: List[str] = []
    used_channels: List[str] = []
    rag_system_prompt: Optional[str] = None
    rag_user_prompt: Optional[str] = None
    model_usage: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = []

    model_config = {"protected_namespaces": ()}