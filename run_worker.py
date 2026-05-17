#!/usr/bin/env python3
"""
文档处理 Worker 启动脚本
从 MinIO 获取文件，解析、切块、向量化，存入 Chroma
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv('.env')

from app.core.document_worker import create_document_worker


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║           Document Processing Worker                      ║
║           文档处理 Worker - Kafka 消费者                  ║
╠═══════════════════════════════════════════════════════════╣
║  功能: 消费 Kafka 消息，异步处理文档                      ║
║  流程: MinIO下载 -> 解析 -> 切块 -> 向量化 -> Chroma存储 ║
╚═══════════════════════════════════════════════════════════╝
    """)

    worker = create_document_worker()
    print("Worker 初始化完成，开始监听 Kafka...")

    try:
        worker.start()
    except KeyboardInterrupt:
        print("\nWorker 已停止")


if __name__ == "__main__":
    main()
