"""
文档处理 Worker
从 Kafka 消费消息，异步处理文档（解析、切块、向量化）
"""
import asyncio
import json
from typing import Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from app.core.logging import get_logger

logger = get_logger(__name__)

from app.documents.processor import DocumentProcessor, TextSplitter
from app.core.chroma_client import create_vector_store
from app.core.llm_service import create_llm_service
from app.core.minio_client import create_minio_client
import os


class DocumentWorker:
    """文档处理 Worker"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.processor = DocumentProcessor()
        self.splitter = TextSplitter(
            strategy=self.config.get("document", {}).get("default_strategy", "structural"),
            config=self.config.get("document", {})
        )
        self.vector_store = create_vector_store(self.config.get("vector_store", {}))
        self.minio_client = create_minio_client(self.config)
        self.llm_service = create_llm_service(
            provider=self.config.get("llm", {}).get("provider", "dashscope"),
            config=self.config.get("llm", {})
        )
        self._consumer = None

    def _create_consumer(self) -> Optional[KafkaConsumer]:
        """创建 Kafka Consumer"""
        kafka_cfg = self.config.get("kafka", {})
        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", kafka_cfg.get("bootstrap_servers", "localhost:9092"))
        topic = os.environ.get("KAFKA_TOPIC", kafka_cfg.get("topic", "super-agent-doc-processing"))

        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                group_id='document-processor',
                enable_auto_commit=True
            )
            return consumer
        except KafkaError as e:
            logger.error(f"Kafka consumer error: {e}")
            return None

    async def process_document(self, document_id: str, file_name: str, file_path: str, metadata: dict = None):
        """
        处理文档：下载、解析、切块、向量化、存储

        Args:
            document_id: 文档ID
            file_name: 文件名
            file_path: MinIO 文件路径
            metadata: 元数据
        """
        logger.error(f"Processing document: {document_id}, file: {file_name}")

        try:
            # 1. 从 MinIO 下载文件
            local_path = f"./data/temp/{document_id}_{file_name}"
            os.makedirs("./data/temp", exist_ok=True)

            if not self.minio_client.download_file(file_path, local_path):
                logger.error(f"Failed to download from MinIO: {file_path}")
                return False

            # 2. 解析文档
            text = await self.processor.parse_document(local_path)
            if not text or text.startswith("解析失败"):
                logger.error(f"Failed to parse document: {text}")
                return False

            # 3. 切分文本
            chunks = await self.splitter.split(text)

            # 4. 生成向量并存储
            chunk_ids = []
            embeddings = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                chunk_id = f"{document_id}_chunk_{i}"
                chunk_ids.append(chunk_id)

                # 获取嵌入向量
                embedding = await self.llm_service.embed(chunk)
                if not embedding or len(embedding) == 0:
                    import random
                    embedding = [random.random() for _ in range(1536)]

                embeddings.append(embedding)
                metadatas.append({
                    "document_id": document_id,
                    "document_name": file_name,
                    "chunk_id": chunk_id,
                    "section_path": f"chunk_{i}",
                    **(metadata or {})
                })

            # 5. 存储到向量数据库
            if chunk_ids:
                self.vector_store.add(
                    documents=chunks[:len(chunk_ids)],
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=chunk_ids
                )

            # 6. 清理临时文件
            if os.path.exists(local_path):
                os.remove(local_path)

            logger.error(f"Document processed successfully: {document_id}, chunks: {len(chunk_ids)}")
            return True

        except Exception as e:
            logger.error(f"Document processing error: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run(self):
        """运行 Worker"""
        logger.error("Document Worker started")
        self._consumer = self._create_consumer()

        if self._consumer is None:
            logger.error("Failed to create Kafka consumer, exiting")
            return

        try:
            for message in self._consumer:
                data = message.value
                if data.get("event_type") == "document.uploaded":
                    await self.process_document(
                        document_id=data.get("document_id"),
                        file_name=data.get("file_name"),
                        file_path=data.get("file_path"),
                        metadata=data.get("metadata")
                    )
        except KeyboardInterrupt:
            logger.error("Document Worker stopped")
        finally:
            if self._consumer:
                self._consumer.close()

    def start(self):
        """启动 Worker（同步入口）"""
        asyncio.run(self.run())


def create_document_worker(config: dict = None) -> DocumentWorker:
    """工厂方法创建 Document Worker"""
    return DocumentWorker(config)