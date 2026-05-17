"""
Neo4j 图谱初始化脚本
用于创建文档的图谱结构：Document -> Chapter -> Section -> Item

使用方式:
    python scripts/init_neo4j_graph.py --document-id 1 --title "示例文档"
    python scripts/init_neo4j_graph.py --demo  # 创建演示数据
"""
import argparse
import sys
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, ".")

from app.core.neo4j_client import Neo4jClient
from app.config import get_settings


class Neo4jGraphInitializer:
    """Neo4j 图谱初始化器"""

    def __init__(self, neo4j_client: Neo4jClient):
        self.client = neo4j_client

    def init_document(self, document_id: int, title: str, metadata: Dict = None) -> bool:
        """
        初始化文档节点

        Args:
            document_id: 文档ID
            title: 文档标题
            metadata: 文档元数据

        Returns:
            是否成功
        """
        return self.client.create_document_node(document_id, title, metadata)

    def add_chapter(self, document_id: int, chapter_title: str, chapter_content: str = "") -> bool:
        """
        添加章节

        Args:
            document_id: 文档ID
            chapter_title: 章节标题
            chapter_content: 章节内容

        Returns:
            是否成功
        """
        return self.client.create_chapter_relationship(document_id, chapter_title, chapter_content)

    def add_section(self, chapter_title: str, section_title: str, section_content: str = "") -> bool:
        """
        添加小节

        Args:
            chapter_title: 所属章节标题
            section_title: 小节标题
            section_content: 小节内容

        Returns:
            是否成功
        """
        return self.client.create_section_relationship(chapter_title, section_title, section_content)

    def add_item(self, section_title: str, item_title: str, item_content: str, item_index: int) -> bool:
        """
        添加条目

        Args:
            section_title: 所属小节标题
            item_title: 条目标题
            item_content: 条目内容
            item_index: 条目索引

        Returns:
            是否成功
        """
        return self.client.create_item_relationship(section_title, item_title, item_content, item_index)

    def create_sample_document(self, document_id: int = 1) -> bool:
        """
        创建示例文档

        Args:
            document_id: 文档ID

        Returns:
            是否成功
        """
        print(f"正在创建示例文档 (ID: {document_id})...")

        # 创建文档节点
        try:
            if not self.client.create_document_node(document_id, "软件架构设计文档"):
                print("创建文档节点失败")
                return False
            print("  文档节点创建成功")
        except Exception as e:
            print(f"  创建文档节点异常: {e}")
            return False

        # 添加章节
        chapters = [
            ("第1章 概述", "本章介绍软件架构的背景和目标"),
            ("第2章 系统架构", "本章描述系统的整体架构设计"),
            ("第3章 模块设计", "本章详细说明各模块的设计"),
            ("第4章 数据模型", "本章介绍核心数据模型"),
            ("第5章 部署方案", "本章说明系统的部署策略"),
        ]

        for chapter_title, chapter_content in chapters:
            print(f"  添加章节: {chapter_title}")
            self.add_chapter(document_id, chapter_title, chapter_content)

            # 为每个章节添加小节
            section_num = chapter_title.split(" ")[0][1:]
            sections = [
                (f"{section_num}.1 背景", "介绍相关背景知识"),
                (f"{section_num}.2 设计原则", "说明设计原则"),
                (f"{section_num}.3 实现方案", "描述实现方案"),
            ]
            for section_title, section_content in sections:
                print(f"    添加小节: {section_title}")
                self.add_section(chapter_title, section_title, section_content)

                # 为每个小节添加条目
                for i in range(1, 4):
                    item_title = f"{section_title.split('.')[1]}.{i} 具体条目{i}"
                    item_content = f"这是 {item_title} 的详细内容..."
                    self.add_item(section_title, item_title, item_content, i)

        print(f"示例文档创建完成！")
        return True

    def clear_document(self, document_id: int) -> bool:
        """
        清除文档的所有图谱数据

        Args:
            document_id: 文档ID

        Returns:
            是否成功
        """
        return self.client.delete_document(document_id)


def main():
    parser = argparse.ArgumentParser(description="Neo4j 图谱初始化工具")
    parser.add_argument("--document-id", type=int, help="文档ID")
    parser.add_argument("--title", type=str, help="文档标题")
    parser.add_argument("--clear", action="store_true", help="清除指定文档的图谱数据")
    parser.add_argument("--demo", action="store_true", help="创建演示数据")

    args = parser.parse_args()

    # 获取配置
    settings = get_settings()

    # 创建 Neo4j 客户端
    client = Neo4jClient(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password
    )

    # 检查连接
    if not client.health_check():
        print("错误: 无法连接到 Neo4j，请检查配置")
        print(f"连接地址: {settings.neo4j.uri}")
        print(f"用户名: {settings.neo4j.username}")
        return 1

    print(f"成功连接到 Neo4j: {settings.neo4j.uri}")

    initializer = Neo4jGraphInitializer(client)

    try:
        if args.clear and args.document_id:
            print(f"正在清除文档 {args.document_id} 的图谱数据...")
            if initializer.clear_document(args.document_id):
                print("清除完成")
            else:
                print("清除失败")
            return 0

        if args.demo:
            initializer.create_sample_document(1)
            return 0

        if args.document_id and args.title:
            print(f"正在创建文档: {args.title} (ID: {args.document_id})")
            initializer.init_document(args.document_id, args.title)
            print("创建完成")
            return 0

        print("请指定操作:")
        print("  --demo              创建演示数据")
        print("  --document-id <id> --title <title>  创建指定文档")
        print("  --clear --document-id <id>  清除指定文档")
        return 1

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())