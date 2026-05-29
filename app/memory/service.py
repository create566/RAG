"""
会话记忆服务
支持：无记忆、滑动窗口、摘要压缩三种策略
"""
import asyncio
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import json


class MemoryStrategy(ABC):
    """记忆策略基类"""

    @abstractmethod
    async def get_context(self, conversation_id: str, history: List[Dict[str, Any]]) -> str:
        """获取记忆上下文"""
        pass

    @abstractmethod
    async def save_exchange(self, conversation_id: str, user_message: str, assistant_message: str):
        """保存对话交换"""
        pass


class NoMemoryStrategy(MemoryStrategy):
    """无记忆策略"""

    async def get_context(self, conversation_id: str, history: List[Dict[str, Any]]) -> str:
        return ""

    async def save_exchange(self, conversation_id: str, user_message: str, assistant_message: str):
        pass  # 不做任何处理


class SlidingWindowMemoryStrategy(MemoryStrategy):
    """滑动窗口记忆策略"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._storage: Dict[str, List[Dict[str, Any]]] = {}

    async def get_context(self, conversation_id: str, history: List[Dict[str, Any]]) -> str:
        if conversation_id not in self._storage:
            self._storage[conversation_id] = []

        # 返回最近N轮对话
        recent = self._storage[conversation_id][-self.window_size:]
        return self._format_context(recent)

    async def save_exchange(self, conversation_id: str, user_message: str, assistant_message: str):
        if conversation_id not in self._storage:
            self._storage[conversation_id] = []

        self._storage[conversation_id].append({
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": datetime.now().isoformat()
        })

        # 保持窗口大小
        if len(self._storage[conversation_id]) > self.window_size * 2:
            self._storage[conversation_id] = self._storage[conversation_id][-self.window_size:]

    def _format_context(self, exchanges: List[Dict[str, Any]]) -> str:
        if not exchanges:
            return ""
        return "\n".join([f"用户: {e['user']}\n助手: {e['assistant']}" for e in exchanges])


class SummaryCompressionMemoryStrategy(MemoryStrategy):
    """摘要压缩记忆策略 - 生产环境推荐"""

    def __init__(self,
                 recent_turns: int = 4,
                 recent_max_chars: int = 2200,
                 summary_max_turns: int = 6,
                 summary_max_chars: int = 1400,
                 llm_service=None):
        self.recent_turns = recent_turns
        self.recent_max_chars = recent_max_chars
        self.summary_max_turns = summary_max_turns
        self.summary_max_chars = summary_max_chars
        self.llm_service = llm_service

        self._storage: Dict[str, Dict[str, Any]] = {}

    async def get_context(self, conversation_id: str, history: List[Dict[str, Any]]) -> str:
        if conversation_id not in self._storage:
            self._storage[conversation_id] = {
                "summary": "",
                "recent": [],
                "exchange_count": 0
            }

        state = self._storage[conversation_id]

        # 构建上下文：长期摘要 + 最近原文窗口
        parts = []
        if state["summary"]:
            parts.append(f"【会话摘要】\n{state['summary']}")
        if state["recent"]:
            parts.append(f"【最近对话】\n" + "\n".join(state["recent"][-self.recent_turns:]))

        return "\n\n".join(parts) if parts else ""

    async def save_exchange(self, conversation_id: str, user_message: str, assistant_message: str):
        if conversation_id not in self._storage:
            self._storage[conversation_id] = {
                "summary": "",
                "recent": [],
                "exchange_count": 0
            }

        state = self._storage[conversation_id]

        # 保存最近对话
        exchange = f"用户: {user_message}\n助手: {assistant_message}"
        state["recent"].append(exchange)
        state["exchange_count"] += 1

        # 截断recent超过预算
        recent_text = "\n\n".join(state["recent"][-self.recent_turns:])
        if len(recent_text) > self.recent_max_chars:
            state["recent"] = self._truncate_recent(state["recent"])

        # 增量摘要：每summary_max_turns轮进行一次压缩
        if state["exchange_count"] % self.summary_max_turns == 0 and self.llm_service:
            new_summary = await self._generate_summary(state["summary"], state["recent"][-self.summary_max_turns:])
            state["summary"] = self._merge_summary(state["summary"], new_summary)

    async def _generate_summary(self, existing_summary: str, recent_exchanges: List[str]) -> str:
        """生成摘要"""
        if not self.llm_service:
            return existing_summary

        prompt = f"""请将以下对话总结为简洁的摘要，保留关键信息：

现有摘要: {existing_summary or '无'}

最近对话:
{chr(10).join(recent_exchanges)}

请返回简洁的摘要（不超过{self.summary_max_chars}字符）："""

        response = await self.llm_service.chat(prompt)
        return response[:self.summary_max_chars]

    def _merge_summary(self, existing: str, new: str) -> str:
        """合并摘要"""
        if not existing:
            return new
        combined = f"{existing}\n{new}"
        return combined[:self.summary_max_chars] if len(combined) > self.summary_max_chars else combined

    def _truncate_recent(self, recent: List[str]) -> List[str]:
        """截断recent"""
        result = []
        char_count = 0
        for exchange in reversed(recent):
            if char_count + len(exchange) <= self.recent_max_chars:
                result.insert(0, exchange)
                char_count += len(exchange)
            else:
                break
        return result


class MySQLMemoryStrategy(MemoryStrategy):
    """MySQL 持久化记忆策略 - 使用 asyncio.to_thread 处理同步 pymysql"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 3306,
                 username: str = "root",
                 password: str = "password",
                 database: str = "super",
                 recent_turns: int = 4,
                 recent_max_chars: int = 2200,
                 summary_max_turns: int = 6,
                 summary_max_chars: int = 1400,
                 llm_service=None,
                 redis_cache=None,
                 embedding_service=None,
                 conversation_memory_store=None):
        import pymysql
        from pymysql.cursors import DictCursor

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.recent_turns = recent_turns
        self.recent_max_chars = recent_max_chars
        self.summary_max_turns = summary_max_turns
        self.summary_max_chars = summary_max_chars
        self.llm_service = llm_service
        self.redis_cache = redis_cache
        self.embedding_service = embedding_service
        self.conversation_memory_store = conversation_memory_store

        self._connection_params = {
            "host": host,
            "port": port,
            "user": username,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "cursorclass": DictCursor
        }

        # 初始化表
        self._init_tables()

    def _get_connection(self):
        import pymysql
        from pymysql.cursors import DictCursor
        return pymysql.connect(**{**self._connection_params, "cursorclass": DictCursor})

    def _init_tables(self):
        """初始化表结构"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # 用户表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100),
                        password_hash VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_username (username)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                # 对话历史表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        conversation_id VARCHAR(255) NOT NULL,
                        user_message TEXT NOT NULL,
                        assistant_message TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user_id (user_id),
                        INDEX idx_conversation_id (conversation_id),
                        INDEX idx_created_at (created_at),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                # 摘要表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_summary (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        conversation_id VARCHAR(255) NOT NULL,
                        summary TEXT,
                        exchange_count INT DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_user_conversation (user_id, conversation_id),
                        INDEX idx_user_id (user_id),
                        INDEX idx_conversation_id (conversation_id),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
            conn.commit()
        finally:
            conn.close()

    async def get_context(self, conversation_id: str, user_id: int, history: List[Dict[str, Any]]) -> str:
        # 先查 Redis 缓存
        if self.redis_cache:
            cached = await self.redis_cache.get_recent(conversation_id, count=self.recent_turns)
            if cached:
                # 从 Redis 构建上下文
                parts = []
                # 从 MySQL 获取摘要
                conn = await asyncio.to_thread(self._get_connection)
                try:
                    with conn.cursor() as cursor:
                        await asyncio.to_thread(cursor.execute,
                            "SELECT summary FROM conversation_summary WHERE conversation_id = %s AND user_id = %s",
                            (conversation_id, user_id)
                        )
                        row = await asyncio.to_thread(cursor.fetchone)
                        if row and row["summary"]:
                            parts.append(f"【会话摘要】\n{row['summary']}")
                finally:
                    conn.close()

                if cached:
                    recent = [f"用户: {m['content']}\n助手: " for m in cached if m.get("role") == "user"]
                    # 重建 recent 对话格式
                    formatted = []
                    for i in range(0, len(cached) - 1, 2):
                        if i + 1 < len(cached):
                            if cached[i].get("role") == "user" and cached[i + 1].get("role") == "assistant":
                                formatted.append(f"用户: {cached[i]['content']}\n助手: {cached[i + 1]['content']}")
                    if formatted:
                        parts.append(f"【最近对话】\n" + "\n".join(formatted[-self.recent_turns:]))
                return "\n\n".join(parts) if parts else ""

        # Redis miss，查 MySQL
        conn = await asyncio.to_thread(self._get_connection)
        try:
            with conn.cursor() as cursor:
                # 获取摘要
                await asyncio.to_thread(cursor.execute,
                    "SELECT summary, exchange_count FROM conversation_summary WHERE conversation_id = %s AND user_id = %s",
                    (conversation_id, user_id)
                )
                summary_row = await asyncio.to_thread(cursor.fetchone)
                summary = summary_row["summary"] if summary_row else ""
                exchange_count = summary_row["exchange_count"] if summary_row else 0

                # 获取最近的对话
                await asyncio.to_thread(cursor.execute, """
                    SELECT user_message, assistant_message
                    FROM conversation_history
                    WHERE conversation_id = %s AND user_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                """, (conversation_id, user_id, self.recent_turns))
                rows = await asyncio.to_thread(cursor.fetchall)
                recent = []
                for row in reversed(rows):
                    recent.append(f"用户: {row['user_message']}\n助手: {row['assistant_message']}")

                # 构建上下文
                parts = []
                if summary:
                    parts.append(f"【会话摘要】\n{summary}")
                if recent:
                    parts.append(f"【最近对话】\n" + "\n".join(recent))

                return "\n\n".join(parts) if parts else ""
        finally:
            conn.close()

    async def save_exchange(self, conversation_id: str, user_id: int, user_message: str, assistant_message: str):
        # 同时写入 Redis 缓存
        if self.redis_cache:
            await self.redis_cache.append_exchange(conversation_id, user_message, assistant_message)

        conn = await asyncio.to_thread(self._get_connection)
        try:
            with conn.cursor() as cursor:
                # 插入对话记录
                await asyncio.to_thread(cursor.execute, """
                    INSERT INTO conversation_history (user_id, conversation_id, user_message, assistant_message)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, conversation_id, user_message, assistant_message))

                # 更新或插入摘要记录
                await asyncio.to_thread(cursor.execute, """
                    INSERT INTO conversation_summary (user_id, conversation_id, summary, exchange_count)
                    VALUES (%s, %s, '', 1)
                    ON DUPLICATE KEY UPDATE exchange_count = exchange_count + 1
                """, (user_id, conversation_id))

                # 获取当前状态
                await asyncio.to_thread(cursor.execute,
                    "SELECT summary, exchange_count FROM conversation_summary WHERE conversation_id = %s AND user_id = %s",
                    (conversation_id, user_id)
                )
                row = await asyncio.to_thread(cursor.fetchone)

                if row and row["exchange_count"] % self.summary_max_turns == 0 and self.llm_service:
                    # 生成新摘要
                    await asyncio.to_thread(cursor.execute, """
                        SELECT user_message, assistant_message
                        FROM conversation_history
                        WHERE conversation_id = %s AND user_id = %s
                        ORDER BY id DESC
                        LIMIT %s
                    """, (conversation_id, user_id, self.summary_max_turns))
                    recent_rows = await asyncio.to_thread(cursor.fetchall)
                    recent_exchanges = []
                    for r in reversed(recent_rows):
                        recent_exchanges.append(f"用户: {r['user_message']}\n助手: {r['assistant_message']}")

                    prompt = f"""请将以下对话总结为简洁的摘要，保留关键信息：

现有摘要: {row['summary'] or '无'}

最近对话:
{chr(10).join(recent_exchanges)}

请返回简洁的摘要（不超过{self.summary_max_chars}字符）："""

                    new_summary = await self.llm_service.chat(prompt)
                    new_summary = new_summary[:self.summary_max_chars]

                    # 合并摘要
                    if row['summary']:
                        combined = f"{row['summary']}\n{new_summary}"
                    else:
                        combined = new_summary
                    combined = combined[:self.summary_max_chars]

                    await asyncio.to_thread(cursor.execute,
                        "UPDATE conversation_summary SET summary = %s WHERE conversation_id = %s AND user_id = %s",
                        (combined, conversation_id, user_id)
                    )

                    # 存入向量库
                    if self.embedding_service and self.conversation_memory_store and combined:
                        await self._save_to_vector_store(combined, conversation_id, user_id, "summary")

            await asyncio.to_thread(conn.commit)
        finally:
            conn.close()

    async def _save_to_vector_store(self, content: str, conversation_id: str, user_id: int, mem_type: str):
        """保存记忆到向量库"""
        try:
            emb = await self.embedding_service.embed(content)
            if emb and len(emb) > 0:
                import uuid
                from datetime import datetime
                self.conversation_memory_store.add(
                    documents=[content],
                    embeddings=[emb],
                    metadatas=[{
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "type": mem_type,
                        "created_at": datetime.now().isoformat()
                    }],
                    ids=[str(uuid.uuid4())]
                )
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"[Memory] Failed to save to vector store: {e}")


class ConversationMemoryService:
    """会话记忆服务"""

    def __init__(self, strategy: MemoryStrategy = None):
        self.strategy = strategy or SummaryCompressionMemoryStrategy()

    async def load_memory_context(self, conversation_id: str, user_id: int = None, trace_recorder=None) -> Dict[str, Any]:
        """加载记忆上下文"""
        if user_id is None:
            context = ""
        else:
            context = await self.strategy.get_context(conversation_id, user_id, [])
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "long_term_summary": "",
            "recent_transcript": context,
            "compression_applied": isinstance(self.strategy, SummaryCompressionMemoryStrategy)
        }

    async def save_exchange(self, conversation_id: str, user_id: int, user_message: str, assistant_message: str):
        """保存对话交换"""
        await self.strategy.save_exchange(conversation_id, user_id, user_message, assistant_message)

    def set_strategy(self, strategy: MemoryStrategy):
        """切换记忆策略"""
        self.strategy = strategy