"""
文档路由服务 - 对标Java的DocumentQuestionRouter
"""
from typing import Dict, Any, Optional
from app.rag.orchestrator import DocumentNavigationDecision, RetrievalQuestionPlan
from app.models.chat import ExecutionMode


class DocumentQuestionRouter:
    """文档问题路由 - 对标Java的DocumentQuestionRouter"""

    def __init__(self, graph_query_engine=None, config: Dict = None):
        self.graph_query_engine = graph_query_engine
        self.config = config or {}

    async def route(self, document_id: Optional[int], question: str, rewrite_result) -> DocumentNavigationDecision:
        """路由到图查询或混合检索 - 对标Java的route方法"""
        if document_id is None:
            return DocumentNavigationDecision(
                execution_mode=ExecutionMode.RETRIEVAL,
                summary_text="无指定文档，使用混合检索"
            )

        # 检查是否需要图查询
        needs_graph = self._check_needs_graph(question)

        if needs_graph and self.graph_query_engine:
            # 图查询模式 - 执行实际图查询
            structure_hint = self._extract_section_hint(question)
            item_index = self._extract_item_index(question)

            # 执行图查询获取结构信息
            graph_result = await self._execute_graph_query(document_id, structure_hint, item_index)

            return DocumentNavigationDecision(
                execution_mode=ExecutionMode.GRAPH_THEN_EVIDENCE,
                retrieval_plan=RetrievalQuestionPlan(
                    retrieval_question=rewrite_result.rewritten_question if rewrite_result else question,
                    sub_questions=rewrite_result.sub_questions if rewrite_result else [question]
                ),
                structure_anchor={
                    "target_section_hint": structure_hint,
                    "graph_data": graph_result
                },
                item_anchor={"item_index": item_index} if item_index else None,
                summary_text=f"文档{document_id}图查询模式 - {graph_result.get('summary', '')}"
            )

        # 默认使用混合检索
        return DocumentNavigationDecision(
            execution_mode=ExecutionMode.RETRIEVAL,
            retrieval_plan=RetrievalQuestionPlan(
                retrieval_question=rewrite_result.rewritten_question if rewrite_result else question,
                sub_questions=rewrite_result.sub_questions if rewrite_result else [question]
            ),
            summary_text=f"文档{document_id}混合检索模式"
        )

    async def _execute_graph_query(self, document_id: int, section_hint: str, item_index: Optional[int] = None) -> Dict[str, Any]:
        """
        执行图查询获取章节结构信息

        Args:
            document_id: 文档ID
            section_hint: 章节提示 (如 "第3章")
            item_index: 可选的条目索引

        Returns:
            图查询结果字典
        """
        if not self.graph_query_engine:
            return {"status": "no_engine", "summary": "图查询引擎未配置"}

        try:
            if item_index is not None:
                # 查询特定条目
                result = await self.graph_query_engine.query_section_content(document_id, section_hint, item_index)
                if result.get("status") == "success":
                    return {
                        "status": "success",
                        "type": "item",
                        "item_title": result.get("item_title", ""),
                        "item_content": result.get("item_content", ""),
                        "summary": f"第{item_index}条: {result.get('item_title', '')}"
                    }
            else:
                # 查询章节内容
                result = await self.graph_query_engine.query_section_content(document_id, section_hint, None)
                if result.get("status") == "success":
                    return {
                        "status": "success",
                        "type": "section",
                        "chapter_title": result.get("chapter_title", ""),
                        "chapter_content": result.get("chapter_content", ""),
                        "sections": result.get("sections", []),
                        "paragraphs": result.get("paragraphs", []),
                        "summary": result.get("chapter_title", section_hint)
                    }

            # 文档结构查询（用于目录类问题）
            structure = await self.graph_query_engine.query_document_structure(document_id)
            if structure.get("status") == "success":
                return {
                    "status": "success",
                    "type": "structure",
                    "document_title": structure.get("document_title", ""),
                    "chapters": structure.get("chapters", []),
                    "sections": structure.get("sections", []),
                    "paragraph_count": structure.get("paragraph_count", 0),
                    "summary": f"《{structure.get('document_title', '')}》包含 {len(structure.get('chapters', []))} 个章节"
                }

            return {"status": "not_found", "summary": f"未找到文档{document_id}的结构信息"}

        except Exception as e:
            return {"status": "error", "summary": f"图查询失败: {str(e)}"}

    def _check_needs_graph(self, question: str) -> bool:
        """检查是否需要图查询"""
        graph_keywords = ["第几章", "章节", "目录", "结构", "哪个部分", "哪些内容", "有哪些章节", "讲了什么", "内容概览"]
        return any(kw in question for kw in graph_keywords)

    def _extract_section_hint(self, question: str) -> str:
        """提取章节提示"""
        import re
        match = re.search(r'[第]([\d]+)[章节]', question)
        if match:
            return match.group(0)
        return ""

    def _extract_item_index(self, question: str) -> Optional[int]:
        """提取条目索引"""
        import re
        match = re.search(r'[第]([\d]+)[条项点]', question)
        if match:
            return int(match.group(1))
        return None