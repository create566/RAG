"""
Neo4j 图数据库客户端封装
对标 Java 的 GraphDatabaseClient
"""
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase


class Neo4jClient:
    """Neo4j 图数据库客户端封装"""

    def __init__(self, uri: str, username: str, password: str):
        """
        初始化 Neo4j 客户端

        Args:
            uri: Neo4j 连接地址 (bolt://localhost:7687)
            username: 用户名
            password: 密码
        """
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None

    @property
    def driver(self):
        """延迟初始化 driver"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
        return self._driver

    def close(self):
        """关闭驱动连接"""
        if self._driver:
            self._driver.close()
            self._driver = None

    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        执行 Cypher 查询

        Args:
            cypher: Cypher 查询语句
            params: 查询参数

        Returns:
            查询结果列表
        """
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def get_document_structure(self, document_id: int) -> Dict[str, Any]:
        """
        获取文档的章节结构

        Args:
            document_id: 文档ID

        Returns:
            包含章节结构的字典
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:CONTAINS]->(chapter:Chapter)
        OPTIONAL MATCH (chapter)-[:CONTAINS]->(section:Section)
        OPTIONAL MATCH (section)-[:CONTAINS]->(paragraph:Paragraph)
        RETURN d.title as document_title,
               collect(DISTINCT chapter.title) as chapters,
               collect(DISTINCT section.title) as sections,
               count(DISTINCT paragraph) as paragraph_count
        """
        results = self.query(cypher, {"document_id": document_id})
        if results:
            return {
                "document_id": document_id,
                "document_title": results[0].get("document_title", ""),
                "chapters": results[0].get("chapters", []),
                "sections": results[0].get("sections", []),
                "paragraph_count": results[0].get("paragraph_count", 0)
            }
        return {"document_id": document_id, "document_title": "", "chapters": [], "sections": [], "paragraph_count": 0}

    def query_section(self, document_id: int, section_hint: str) -> List[Dict[str, Any]]:
        """
        根据章节提示查询章节内容

        Args:
            document_id: 文档ID
            section_hint: 章节提示 (如 "第3章")

        Returns:
            匹配的章节列表
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})-[:CONTAINS]->(chapter:Chapter)
        WHERE chapter.title CONTAINS $section_hint
        OPTIONAL MATCH (chapter)-[:CONTAINS]->(section:Section)
        OPTIONAL MATCH (section)-[:CONTAINS]->(paragraph:Paragraph)
        RETURN chapter.title as chapter_title,
               chapter.content as chapter_content,
               collect(DISTINCT section.title) as sections,
               collect(DISTINCT paragraph.content) as paragraphs
        """
        results = self.query(cypher, {"document_id": document_id, "section_hint": section_hint})
        return results

    def query_item_in_section(self, document_id: int, section_hint: str, item_index: int) -> List[Dict[str, Any]]:
        """
        查询章节中的特定条目 (第X条/第X点)

        Args:
            document_id: 文档ID
            section_hint: 章节提示
            item_index: 条目索引

        Returns:
            匹配的条目列表
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})-[:CONTAINS]->(chapter:Chapter)
        WHERE chapter.title CONTAINS $section_hint
        MATCH (chapter)-[:CONTAINS]->(item:Item)
        WHERE item.index = $item_index
        RETURN item.title as item_title,
               item.content as item_content,
               item.index as item_index
        """
        results = self.query(cypher, {"document_id": document_id, "section_hint": section_hint, "item_index": item_index})
        return results

    def create_document_node(self, document_id: int, title: str, metadata: Optional[Dict] = None) -> bool:
        """
        创建文档节点

        Args:
            document_id: 文档ID
            title: 文档标题
            metadata: 文档元数据

        Returns:
            是否创建成功
        """
        import json
        cypher = """
        CREATE (d:Document {
            document_id: $document_id,
            title: $title,
            metadata: $metadata,
            created_at: datetime()
        })
        RETURN d.document_id as id
        """
        try:
            results = self.query(cypher, {
                "document_id": document_id,
                "title": title,
                "metadata": json.dumps(metadata) if metadata else "{}"
            })
            return bool(results)
        except Exception as e:
            print(f"Neo4j create_document_node error: {e}")
            return False

    def create_chapter_relationship(self, document_id: int, chapter_title: str, chapter_content: str = "") -> bool:
        """
        创建文档与章节的关系

        Args:
            document_id: 文档ID
            chapter_title: 章节标题
            chapter_content: 章节内容

        Returns:
            是否创建成功
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})
        CREATE (d)-[:CONTAINS]->(c:Chapter {
            title: $chapter_title,
            content: $chapter_content,
            created_at: datetime()
        })
        RETURN c.title as title
        """
        try:
            results = self.query(cypher, {
                "document_id": document_id,
                "chapter_title": chapter_title,
                "chapter_content": chapter_content
            })
            return bool(results)
        except Exception:
            return False

    def create_section_relationship(self, chapter_title: str, section_title: str, section_content: str = "") -> bool:
        """
        创建章节与小节的关系

        Args:
            chapter_title: 章节标题
            section_title: 小节标题
            section_content: 小节内容

        Returns:
            是否创建成功
        """
        cypher = """
        MATCH (c:Chapter {title: $chapter_title})
        CREATE (c)-[:CONTAINS]->(s:Section {
            title: $section_title,
            content: $section_content,
            created_at: datetime()
        })
        RETURN s.title as title
        """
        try:
            results = self.query(cypher, {
                "chapter_title": chapter_title,
                "section_title": section_title,
                "section_content": section_content
            })
            return bool(results)
        except Exception:
            return False

    def create_item_relationship(self, section_title: str, item_title: str, item_content: str, item_index: int) -> bool:
        """
        创建小节与条目的关系

        Args:
            section_title: 小节标题
            item_title: 条目标题
            item_content: 条目内容
            item_index: 条目索引

        Returns:
            是否创建成功
        """
        cypher = """
        MATCH (s:Section {title: $section_title})
        CREATE (s)-[:CONTAINS]->(i:Item {
            title: $item_title,
            content: $item_content,
            index: $item_index,
            created_at: datetime()
        })
        RETURN i.title as title
        """
        try:
            results = self.query(cypher, {
                "section_title": section_title,
                "item_title": item_title,
                "item_content": item_content,
                "item_index": item_index
            })
            return bool(results)
        except Exception:
            return False

    def create_paragraph_relationship(self, chapter_title: str, paragraph_content: str, paragraph_index: int) -> bool:
        """
        创建章节与段落的关联

        Args:
            chapter_title: 章节标题
            paragraph_content: 段落内容
            paragraph_index: 段落索引

        Returns:
            是否创建成功
        """
        cypher = """
        MATCH (c:Chapter {title: $chapter_title})
        CREATE (c)-[:CONTAINS]->(p:Paragraph {
            content: $paragraph_content,
            index: $paragraph_index,
            created_at: datetime()
        })
        RETURN p.content as content
        """
        try:
            results = self.query(cypher, {
                "chapter_title": chapter_title,
                "paragraph_content": paragraph_content,
                "paragraph_index": paragraph_index
            })
            return bool(results)
        except Exception:
            return False

    def delete_document(self, document_id: int) -> bool:
        """
        删除文档及其所有关联节点

        Args:
            document_id: 文档ID

        Returns:
            是否删除成功
        """
        cypher = """
        MATCH (d:Document {document_id: $document_id})
        DETACH DELETE d
        """
        try:
            self.query(cypher, {"document_id": document_id})
            return True
        except Exception:
            return False

    def health_check(self) -> bool:
        """
        检查 Neo4j 连接健康状态

        Returns:
            是否连接正常
        """
        try:
            self.query("RETURN 1 as health", {})
            return True
        except Exception:
            return False