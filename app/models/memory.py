"""
记忆相关的数据模型
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemorySummary(BaseModel):
    """记忆摘要"""
    conversation_goal: str = ""
    stable_facts: List[str] = []
    pending_questions: List[str] = []
    retrieval_hints: List[str] = []


class MemoryContext(BaseModel):
    """记忆上下文"""
    conversation_id: str
    long_term_summary: str = ""
    recent_transcript: str = ""
    answer_recent_transcript: str = ""
    compression_applied: bool = False
    covered_exchange_id: Optional[int] = None
    covered_exchange_count: int = 0
    compression_count: int = 0
    summary_payload: Optional[MemorySummary] = None


class ConversationMemory(BaseModel):
    """会话记忆"""
    id: Optional[int] = None
    conversation_id: str
    exchange_id: int
    user_message: str
    assistant_message: str
    created_at: datetime = Field(default_factory=datetime.now)