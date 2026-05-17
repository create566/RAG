"""
Kafka 消息队列客户端
用于异步触发文档处理流程
"""
from typing import Optional, Dict, Any
import json
import os
from kafka import KafkaProducer
from kafka.errors import KafkaError

from app.core.logging import get_logger

logger = get_logger(__name__)


class KafkaClient:
    """Kafka 消息队列客户端"""

    def __init__(self, bootstrap_servers: str, topic: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer = None

    @property
    def producer(self) -> Optional[KafkaProducer]:
        """获取 Kafka Producer"""
        if self._producer is None:
            try:
                self._producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks='all',
                    retries=3
                )
            except KafkaError as e:
                logger.error(f"Kafka producer error: {e}")
                return None
        return self._producer

    def send_message(self, message: Dict[str, Any], key: str = None) -> bool:
        """
        发送消息到 Kafka

        Args:
            message: 消息内容（字典）
            key: 消息键（可选）

        Returns:
            是否发送成功
        """
        try:
            if self.producer is None:
                logger.warning("Kafka producer not available")
                return False

            future = self.producer.send(
                self.topic,
                value=message,
                key=key
            )
            # 等待发送完成（同步）
            future.get(timeout=10)
            return True
        except KafkaError as e:
            logger.error(f"Kafka send error: {e}")
            return False

    def send_document_upload_event(self, document_id: str, file_name: str, file_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        发送文档上传事件

        Args:
            document_id: 文档ID
            file_name: 文件名
            file_path: MinIO 中的文件路径
            metadata: 附加元数据

        Returns:
            是否发送成功
        """
        event = {
            "event_type": "document.uploaded",
            "document_id": document_id,
            "file_name": file_name,
            "file_path": file_path,
            "metadata": metadata or {}
        }
        return self.send_message(event, key=document_id)

    def close(self):
        """关闭 producer"""
        if self._producer:
            self._producer.close()
            self._producer = None


def create_kafka_client(config: dict = None) -> Optional[KafkaClient]:
    """工厂方法创建 Kafka 客户端"""
    config = config or {}
    kafka_cfg = config.get("kafka", {})

    # 从环境变量加载（如果有）
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", kafka_cfg.get("bootstrap_servers", "localhost:9092"))
    topic = os.environ.get("KAFKA_TOPIC", kafka_cfg.get("topic", "super-agent-doc-processing"))

    try:
        return KafkaClient(bootstrap_servers, topic)
    except Exception as e:
        logger.error(f"Kafka client creation error: {e}")
        return None