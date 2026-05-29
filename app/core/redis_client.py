"""
Redis 缓存客户端
用于会话热数据的缓存加速
"""
from typing import List, Dict, Any, Optional
import json
import redis.asyncio as redis

from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Redis 缓存客户端"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: str = None, prefix: str = "super_agent:"):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self._client = None

    @property
    def client(self):
        """获取 Redis 客户端"""
        if self._client is None:
            from redis.asyncio import ConnectionPool
            pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=None,
                decode_responses=True,
                max_connections=10
            )
            self._client = redis.Redis(connection_pool=pool)
        return self._client

    def _key(self, conversation_id: str) -> str:
        """生成带前缀的 key"""
        return f"{self.prefix}conversation:{conversation_id}"

    async def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """获取最近 N 轮对话历史"""
        key = self._key(conversation_id)
        data = await self.client.lrange(key, -limit * 2, -1)
        result = []
        for item in data:
            try:
                result.append(json.loads(item))
            except:
                continue
        return result

    async def append_message(self, conversation_id: str, role: str, content: str):
        """追加单条消息到对话历史"""
        key = self._key(conversation_id)
        message = json.dumps({"role": role, "content": content})
        await self.client.rpush(key, message)
        await self.client.expire(key, 86400 * 7)  # 7天过期

    async def append_exchange(self, conversation_id: str, user_message: str, assistant_message: str):
        """追加一轮对话"""
        key = self._key(conversation_id)
        user_msg = json.dumps({"role": "user", "content": user_message})
        asst_msg = json.dumps({"role": "assistant", "content": assistant_message})
        await self.client.rpush(key, user_msg, asst_msg)
        await self.client.expire(key, 86400 * 7)

    async def get_recent(self, conversation_id: str, count: int = 4) -> List[Dict[str, str]]:
        """获取最近 N 轮对话（每轮包含 user + assistant）"""
        messages = await self.get_conversation_history(conversation_id, limit=count * 2)
        return messages[-count * 2:] if messages else []

    async def clear_conversation(self, conversation_id: str):
        """清空对话缓存"""
        key = self._key(conversation_id)
        await self.client.delete(key)

    async def set_summary(self, conversation_id: str, summary: str):
        """缓存会话摘要"""
        key = f"{self.prefix}summary:{conversation_id}"
        await self.client.set(key, summary, ex=86400 * 7)

    async def get_summary(self, conversation_id: str) -> Optional[str]:
        """获取会话摘要"""
        key = f"{self.prefix}summary:{conversation_id}"
        return await self.client.get(key)

    async def close(self):
        """关闭连接"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ========== Redis Stream (异步文档处理队列) ==========

    async def xadd(self, stream: str, message: Dict[str, str], max_len: int = 1000) -> str:
        """添加消息到 Stream"""
        key = f"{self.prefix}stream:{stream}"
        fields = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in message.items()}
        message_id = await self.client.xadd(key, fields, maxlen=max_len)
        return message_id

    async def xread_group(self, group: str, consumer: str, streams: List[str], count: int = 10, block: int = 5000) -> List[Dict]:
        """读取 Stream 消息（群体消费模式）"""
        stream_dict = {f"{self.prefix}stream:{s}": ">" for s in streams}
        try:
            result = await self.client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams=stream_dict,
                count=count,
                block=block
            )
            messages = []
            if result:
                for stream_messages in result:
                    stream_name = stream_messages[0]
                    for msg_id, fields in stream_messages[1]:
                        decoded = {}
                        for k, v in fields.items():
                            try:
                                decoded[k] = json.loads(v) if isinstance(v, str) else v
                            except:
                                decoded[k] = v
                        messages.append({
                            "id": msg_id,
                            "stream": stream_name.replace(f"{self.prefix}stream:", ""),
                            "data": decoded
                        })
            return messages
        except Exception as e:
            logger.warning(f"xread_group failed: {e}")
            return []

    async def xack(self, stream: str, group: str, *message_ids: str) -> int:
        """确认消息已处理"""
        key = f"{self.prefix}stream:{stream}"
        return await self.client.xack(key, group, *message_ids)

    async def xgroup_create(self, stream: str, group: str, id: str = "0", mkstream: bool = True) -> bool:
        """创建消费者群体"""
        key = f"{self.prefix}stream:{stream}"
        try:
            await self.client.xgroup_create(key, group, id, mkstream=mkstream)
            return True
        except Exception as e:
            if "BUSYGROUP" in str(e):
                return True  # 群体已存在
            logger.warning(f"xgroup_create failed: {e}")
            return False


def create_redis_cache(config: Dict = None) -> RedisCache:
    """工厂方法创建 Redis 缓存"""
    config = config or {}
    return RedisCache(
        host=config.get("host", "localhost"),
        port=config.get("port", 6379),
        db=config.get("db", 0),
        password=config.get("password"),
        prefix=config.get("prefix", "super_agent:")
    )