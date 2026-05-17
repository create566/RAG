"""
API路由层
"""
from .chat import router as chat_router
from .document import router as document_router
from .auth import router as auth_router

__all__ = ["chat_router", "document_router", "auth_router"]