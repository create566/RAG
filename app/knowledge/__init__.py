"""
知识模块
"""
from .route import DocumentQuestionRouter
from .service import KnowledgeRouteService
from .document import DocumentService

__all__ = ["DocumentQuestionRouter", "KnowledgeRouteService", "DocumentService"]