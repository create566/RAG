"""
Elasticsearch 索引初始化脚本
用于创建关键词检索索引

使用方式:
    python scripts/init_elasticsearch.py
    python scripts/init_elasticsearch.py --index custom_index
"""
import argparse
import sys

sys.path.insert(0, ".")

from elasticsearch import Elasticsearch
from app.config import get_settings


def create_index(es_client, index_name: str):
    """创建索引并设置映射"""
    mappings = {
        "properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "title": {"type": "text", "analyzer": "standard"},
            "chunk_id": {"type": "keyword"},
            "parent_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "document_name": {"type": "text"},
            "section_path": {"type": "text"},
            "tags": {"type": "text"}
        }
    }

    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }

    if es_client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在，跳过创建")
        return True

    try:
        es_client.indices.create(
            index=index_name,
            mappings=mappings,
            settings=settings
        )
        print(f"索引 '{index_name}' 创建成功")
        return True
    except Exception as e:
        print(f"索引创建失败: {e}")
        return False


def insert_sample_data(es_client, index_name: str):
    """插入示例数据"""
    sample_docs = [
        {
            "content": "这是第一章概述部分的内容，介绍项目背景和目标。",
            "title": "第1章 概述",
            "chunk_id": "ch1_001",
            "parent_id": "ch1",
            "document_id": "1",
            "document_name": "软件架构设计文档",
            "section_path": "第1章 > 1.1 背景",
            "tags": "概述 背景 项目目标"
        },
        {
            "content": "第二章介绍系统架构设计，包括整体架构图和模块划分。",
            "title": "第2章 系统架构",
            "chunk_id": "ch2_001",
            "parent_id": "ch2",
            "document_id": "1",
            "document_name": "软件架构设计文档",
            "section_path": "第2章 > 2.1 整体架构",
            "tags": "架构 模块 设计"
        },
        {
            "content": "第三章详细说明各核心模块的实现细节。",
            "title": "第3章 实现细节",
            "chunk_id": "ch3_001",
            "parent_id": "ch3",
            "document_id": "1",
            "document_name": "软件架构设计文档",
            "section_path": "第3章 > 3.1 核心模块",
            "tags": "实现 模块 细节"
        }
    ]

    print(f"正在插入 {len(sample_docs)} 条示例数据...")
    for i, doc in enumerate(sample_docs):
        try:
            es_client.index(index=index_name, id=i + 1, document=doc)
            print(f"  插入文档 {i + 1}: {doc['title']}")
        except Exception as e:
            print(f"  插入失败: {e}")
            return False

    es_client.indices.refresh(index=index_name)
    print("示例数据插入完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="Elasticsearch 索引初始化工具")
    parser.add_argument("--index", type=str, default="super_agent_keywords", help="索引名称")
    parser.add_argument("--sample", action="store_true", help="插入示例数据")

    args = parser.parse_args()

    settings = get_settings()
    es_hosts = settings.elasticsearch.hosts

    print(f"正在连接 Elasticsearch: {es_hosts}")

    try:
        es_client = Elasticsearch(es_hosts)

        if not es_client.ping():
            print("错误: 无法连接到 Elasticsearch")
            return 1

        print("成功连接到 Elasticsearch")

        if create_index(es_client, args.index):
            if args.sample:
                insert_sample_data(es_client, args.index)

        return 0

    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())