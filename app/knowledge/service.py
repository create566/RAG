"""
知识路由服务 - 对标Java的KnowledgeRouteService
"""
from typing import List, Dict, Any, Optional
from app.models.document import DocumentRouteCandidate, KnowledgeRouteDecision, KnowledgeDomain


class KnowledgeRouteService:
    """知识路由服务"""

    def __init__(self, llm_service, document_service, config: Dict = None):
        self.llm_service = llm_service
        self.document_service = document_service
        self.config = config or {}

    async def route(self, question: str, rewrite_question: str) -> KnowledgeRouteDecision:
        """执行知识路由 - 对标Java的route方法"""
        # 获取可检索文档列表（已按 document_name 去重）
        documents = await self.document_service.list_retrievable_documents()

        if not documents:
            return KnowledgeRouteDecision(
                confidence=0.0,
                route_status="no_documents",
                documents=[]
            )

        # 如果只有一个文档，直接返回，无需 LLM 路由
        if len(documents) == 1:
            doc = documents[0]
            return KnowledgeRouteDecision(
                confidence=0.9,
                route_status="single_document",
                documents=[
                    DocumentRouteCandidate(
                        document_id=str(doc.get("id", "")),
                        document_name=doc.get("document_name", ""),
                        last_index_task_id=str(doc.get("last_index_task_id", "")),
                        knowledge_scope_code=doc.get("knowledge_scope_code", ""),
                        knowledge_scope_name=doc.get("knowledge_scope_name", ""),
                        business_category=doc.get("business_category", ""),
                        document_tags=doc.get("document_tags", ""),
                        score=0.9,
                        reason="唯一文档"
                    )
                ]
            )

        # 使用LLM进行路由决策
        routing_prompt = f"""分析用户问题，确定最相关的文档。

用户问题: {question}
改写后问题: {rewrite_question}

候选文档:
{self._format_documents(documents)}

请分析并返回JSON格式的路由决策：
{{
    "confidence": 0.0-1.0,
    "selected_documents": ["文档1", "文档2"],
    "reason": "选择原因"
}}
"""
        response = await self.llm_service.chat(routing_prompt)

        # 解析响应
        try:
            import json
            result = json.loads(response)
            selected_names = result.get("selected_documents", [])
            confidence = result.get("confidence", 0.5)

            selected_docs = [d for d in documents if d.get("document_name") in selected_names]

            # 如果没有匹配，按相关性排序
            if not selected_docs:
                scored_docs = self._score_documents(question, documents)
                selected_docs = scored_docs[:3]
                confidence = 0.5

            # 去重：确保不返回同名文档
            seen_names = set()
            unique_docs = []
            for doc in selected_docs:
                name = doc.get("document_name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    unique_docs.append(doc)

            candidates = [
                DocumentRouteCandidate(
                    document_id=str(doc.get("id", "")),
                    document_name=doc.get("document_name", ""),
                    last_index_task_id=str(doc.get("last_index_task_id", "")),
                    knowledge_scope_code=doc.get("knowledge_scope_code", ""),
                    knowledge_scope_name=doc.get("knowledge_scope_name", ""),
                    business_category=doc.get("business_category", ""),
                    document_tags=doc.get("document_tags", ""),
                    score=0.8 - i * 0.1,
                    reason="LLM路由选择"
                )
                for i, doc in enumerate(unique_docs[:5])
            ]

            return KnowledgeRouteDecision(
                confidence=confidence,
                route_status="success",
                documents=candidates
            )
        except:
            # 降级：返回前几个文档（去重）
            seen_names = set()
            unique_docs = []
            for doc in documents:
                name = doc.get("document_name", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    unique_docs.append(doc)
                    if len(unique_docs) >= 3:
                        break

            candidates = [
                DocumentRouteCandidate(
                    document_id=str(doc.get("id", "")),
                    document_name=doc.get("document_name", ""),
                    last_index_task_id=str(doc.get("last_index_task_id", "")),
                    knowledge_scope_code=doc.get("knowledge_scope_code", ""),
                    knowledge_scope_name=doc.get("knowledge_scope_name", ""),
                    business_category=doc.get("business_category", ""),
                    document_tags=doc.get("document_tags", ""),
                    score=0.5,
                    reason="降级选择"
                )
                for doc in unique_docs
            ]

            return KnowledgeRouteDecision(
                confidence=0.3,
                route_status="fallback",
                documents=candidates
            )

    def _format_documents(self, documents: List[Dict]) -> str:
        """格式化文档列表（去重）"""
        lines = []
        seen = set()
        for doc in documents[:10]:
            name = doc.get('document_name', '')
            if name and name not in seen:
                seen.add(name)
                lines.append(f"- {name} ({doc.get('knowledge_scope_name', '')})")
        return "\n".join(lines)

    def _score_documents(self, question: str, documents: List[Dict]) -> List[Dict]:
        """对文档进行相关性评分"""
        # 简单基于关键词匹配评分
        question_lower = question.lower()
        scored = []

        for doc in documents:
            score = 0
            name = doc.get("document_name", "").lower()
            scope = doc.get("knowledge_scope_name", "").lower()
            tags = doc.get("document_tags", "").lower()

            for kw in question_lower.split():
                if kw in name:
                    score += 3
                if kw in scope:
                    score += 2
                if kw in tags:
                    score += 1

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]

    async def record_shadow_route(self, conversation_id: str, exchange_id: int, document_id: int, question: str, rewrite_question: str):
        """记录影子路由 - 对标Java的recordShadowRoute"""
        pass

    async def record_auto_route(self, conversation_id: str, exchange_id: int, question: str, rewrite_question: str, route_decision: KnowledgeRouteDecision):
        """记录自动路由 - 对标Java的recordAutoRoute"""
        pass