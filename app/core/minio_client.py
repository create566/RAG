"""
MinIO 对象存储客户端
用于文档文件的云存储
"""
import asyncio
from typing import Optional
from minio import Minio
from minio.error import S3Error
import io
import os

from app.core.logging import get_logger

logger = get_logger(__name__)


class MinioClient:
    """MinIO 对象存储客户端"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure
        self._client = None

    @property
    def client(self) -> Minio:
        """获取 MinIO 客户端"""
        if self._client is None:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
        return self._client

    def ensure_bucket(self) -> bool:
        """确保 bucket 存在"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            return True
        except S3Error as e:
            logger.error(f"bucket error: {e}")
            return False

    def upload_file(self, object_name: str, file_path: str, content_type: str = "application/octet-stream") -> Optional[str]:
        """
        上传文件到 MinIO

        Args:
            object_name: 对象名称
            file_path: 本地文件路径
            content_type: 内容类型

        Returns:
            对象URL，失败返回None
        """
        try:
            self.ensure_bucket()
            self.client.fput_object(
                self.bucket,
                object_name,
                file_path,
                content_type=content_type
            )
            # 返回对象路径
            return f"{self.bucket}/{object_name}"
        except S3Error as e:
            logger.error(f"upload error: {e}")
            return None

    def upload_bytes(self, object_name: str, data: bytes, content_type: str = "application/octet-stream") -> Optional[str]:
        """
        上传字节数据到 MinIO

        Args:
            object_name: 对象名称
            data: 字节数据
            content_type: 内容类型

        Returns:
            对象路径，失败返回None
        """
        try:
            self.ensure_bucket()
            self.client.put_object(
                self.bucket,
                object_name,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type
            )
            return f"{self.bucket}/{object_name}"
        except S3Error as e:
            logger.error(f"upload error: {e}")
            return None

    def download_file(self, object_name: str, file_path: str) -> bool:
        """下载文件到本地"""
        try:
            self.client.fget_object(self.bucket, object_name, file_path)
            return True
        except S3Error as e:
            logger.error(f"download error: {e}")
            return False

    def get_presigned_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """获取预签名URL（用于临时访问）"""
        try:
            return self.client.presigned_get_object(self.bucket, object_name, expires)
        except S3Error as e:
            logger.error(f"presigned URL error: {e}")
            return None

    def delete_file(self, object_name: str) -> bool:
        """删除文件"""
        try:
            self.client.remove_object(self.bucket, object_name)
            return True
        except S3Error as e:
            logger.error(f"delete error: {e}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """列出文件"""
        try:
            objects = self.client.list_objects(self.bucket, prefix=prefix)
            return [obj.object_name for obj in objects]
        except S3Error as e:
            logger.error(f"list error: {e}")
            return []

    # ── 异步包装方法 ───────────────────────────────────────

    async def async_upload_file(self, object_name: str, file_path: str, content_type: str = "application/octet-stream") -> Optional[str]:
        """异步上传文件"""
        return await asyncio.to_thread(self.upload_file, object_name, file_path, content_type)

    async def async_download_file(self, object_name: str, file_path: str) -> bool:
        """异步下载文件"""
        return await asyncio.to_thread(self.download_file, object_name, file_path)

    async def async_get_presigned_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """异步获取预签名URL"""
        return await asyncio.to_thread(self.get_presigned_url, object_name, expires)


def create_minio_client(config: dict = None) -> MinioClient:
    """工厂方法创建 MinIO 客户端"""
    config = config or {}
    minio_cfg = config.get("minio", {})

    # 从环境变量加载（如果有）
    def get_env(key, default=""):
        return os.environ.get(key, minio_cfg.get(key.lower(), default))

    return MinioClient(
        endpoint=get_env("MINIO_ENDPOINT", "localhost:9000"),
        access_key=get_env("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=get_env("MINIO_SECRET_KEY", "minioadmin"),
        bucket=get_env("MINIO_BUCKET", "super-agent"),
        secure=get_env("MINIO_SECURE", "false").lower() == "true"
    )