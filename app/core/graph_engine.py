"""
图查询引擎模块
定义图查询接口和 Neo4j 实现
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from app.core.neo4j_client import Neo4jClient


class BaseGraphQueryEngine(ABC):
    """图查询引擎抽象基类"""

    @abstractmethod
    async def query_document_structure(self, document_id: int) -> Dict[str, Any]:
        """
        查询文档的章节结构

        Args:
            document_id: 文档ID

        Returns:
            包含章节结构的字典
        """
        pass

    @abstractmethod
    async def query_section_content(self, document_id: int, section_hint: str, item_index: Optional[int] = None) -> Dict[str, Any]:
        """
        查询特定章节的内容

        Args:
            document_id: 文档ID
            section_hint: 章节提示 (如 "第3章")
            item_index: 可选的条目索引

        Returns:
            章节内容和相关条目
        """
        pass


class Neo4jGraphQueryEngine(BaseGraphQueryEngine):
    """基于 Neo4j 的图查询引擎"""

    def __init__(self, neo4j_client: Neo4jClient):
        """
        初始化图查询引擎

        Args:
            neo4j_client: Neo4j 客户端实例
        """
        self.client = neo4j_client

    async def query_document_structure(self, document_id: int) -> Dict[str, Any]:
        """
        获取文档的完整章节结构

        Args:
            document_id: 文档ID

        Returns:
            包含文档标题、章节列表、段落数量的字典
        """
        result = await self.client.async_get_document_structure(document_id)
        return {
            "document_id": document_id,
            "document_title": result.get("document_title", ""),
            "chapters": result.get("chapters", []),
            "sections": result.get("sections", []),
            "paragraph_count": result.get("paragraph_count", 0),
            "status": "success" if result.get("document_title") else "not_found"
        }

    async def query_section_content(self, document_id: int, section_hint: str, item_index: Optional[int] = None) -> Dict[str, Any]:
        """
        查询特定章节的内容

        Args:
            document_id: 文档ID
            section_hint: 章节提示 (如 "第3章")
            item_index: 可选的条目索引

        Returns:
            章节内容和条目列表
        """
        if item_index is not None:
            items = await self.client.async_query_item_in_section(document_id, section_hint, item_index)
            if items:
                return {
                    "document_id": document_id,
                    "section_hint": section_hint,
                    "item_index": item_index,
                    "item_title": items[0].get("item_title", ""),
                    "item_content": items[0].get("item_content", ""),
                    "status": "success"
                }

        sections = await self.client.async_query_section(document_id, section_hint)
        if sections:
            section = sections[0]
            return {
                "document_id": document_id,
                "section_hint": section_hint,
                "chapter_title": section.get("chapter_title", ""),
                "chapter_content": section.get("chapter_content", ""),
                "sections": section.get("sections", []),
                "paragraphs": section.get("paragraphs", []),
                "status": "success"
            }

        return {
            "document_id": document_id,
            "section_hint": section_hint,
            "status": "not_found",
            "message": f"未找到章节: {section_hint}"
        }

    async def query_chapter_summary(self, document_id: int) -> List[Dict[str, Any]]:
        """
        获取文档所有章节的摘要

        Args:
            document_id: 文档ID

        Returns:
            章节摘要列表
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})-[:CONTAINS]->(chapter:Chapter)
        RETURN chapter.title as title,
               chapter.content as content,
               size((chapter)-[:CONTAINS]->()) as child_count
        """
        results = await self.client.async_query(cypher, {"document_id": document_id})
        return [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "child_count": r.get("child_count", 0)
            }
            for r in results
        ]

    async def query_table_of_contents(self, document_id: int) -> Dict[str, Any]:
        """
        获取文档目录结构

        Args:
            document_id: 文档ID

        Returns:
            目录结构字典
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH path = (d)-[:CONTAINS*2]->(node)
        WHERE node:Chapter OR node:Section OR node:Item
        RETURN d.title as document_title,
               [(node in nodes(path) | {title: node.title, type: labels(node)[0]})] as toc
        """
        results = await self.client.async_query(cypher, {"document_id": document_id})
        if results:
            return {
                "document_title": results[0].get("document_title", ""),
                "toc": results[0].get("toc", [])
            }
        return {"document_title": "", "toc": []}


class MockGraphQueryEngine(BaseGraphQueryEngine):
    """模拟图查询引擎 (用于测试或 Neo4j 不可用时)"""

    def __init__(self):
        self._mock_data = {
            1: {
                "document_title": "示例文档",
                "chapters": ["第1章 概述", "第2章 架构设计", "第3章 实现细节"],
                "sections": ["1.1 背景", "2.1 系统架构", "3.1 核心模块"],
                "paragraph_count": 50
            }
        }

    async def query_document_structure(self, document_id: int) -> Dict[str, Any]:
        data = self._mock_data.get(document_id, {})
        return {
            "document_id": document_id,
            "document_title": data.get("document_title", ""),
            "chapters": data.get("chapters", []),
            "sections": data.get("sections", []),
            "paragraph_count": data.get("paragraph_count", 0),
            "status": "success" if data else "not_found"
        }

    async def query_section_content(self, document_id: int, section_hint: str, item_index: Optional[int] = None) -> Dict[str, Any]:
        chapters = self._mock_data.get(document_id, {}).get("chapters", [])
        matching = [c for c in chapters if section_hint in c]

        if matching:
            return {
                "document_id": document_id,
                "section_hint": section_hint,
                "chapter_title": matching[0],
                "chapter_content": f"这是 {matching[0]} 的内容描述...",
                "status": "success"
            }

        return {
            "document_id": document_id,
            "section_hint": section_hint,
            "status": "not_found"
        }


def create_graph_engine(provider: str = "neo4j", config: Dict = None) -> BaseGraphQueryEngine:
    """
    工厂方法创建图查询引擎

    Args:
        provider: 图数据库提供者 (neo4j/mock)
        config: 配置字典

    Returns:
        图查询引擎实例
    """
    config = config or {}

    if provider == "neo4j":
        from app.config import get_settings
        settings = get_settings()

        client = Neo4jClient(
            uri=settings.neo4j.uri,
            username=settings.neo4j.username,
            password=settings.neo4j.password
        )

        if not client.health_check():
            raise RuntimeError("Neo4j 连接失败，请检查配置")

        return Neo4jGraphQueryEngine(client)

    elif provider == "mock":
        return MockGraphQueryEngine()

    else:
        raise ValueError(f"不支持的图数据库 provider: {provider}")