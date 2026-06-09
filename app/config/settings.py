"""
统一配置管理 — Pydantic Settings 作为唯一配置源

加载顺序:
1. 读取 .env 文件到环境变量
2. config.yaml 填充结构体默认值
3. 环境变量覆盖（.env 优先于 YAML）
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from app.utils.env import resolve_env, load_yaml_config


# ── 子模型 ──────────────────────────────────────────────

class MySQLConfig(BaseModel):
    host: str = "localhost"
    port: int = 3306
    username: str = "root"
    password: str = "password"
    database: str = "super"

    @property
    def url(self) -> str:
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        return f"mysql+aiomysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class ChromaConfig(BaseModel):
    persist_directory: str = "./data/chroma"
    collection_name: str = "super_agent_docs"


class ElasticsearchConfig(BaseModel):
    hosts: List[str] = ["http://localhost:9200"]
    index: str = "super_agent_keywords"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"


class MinioConfig(BaseModel):
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "super-agent"
    secure: bool = False


class KafkaConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    topic: str = "super-agent-doc-processing"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    prefix: str = "super_agent:"


class LLMConfig(BaseModel):
    provider: str = "dashscope"
    api_key: str = ""
    model: str = "qwen-max"
    max_tokens: int = 4096
    temperature: float = 0.7
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"


class OllamaConfig(BaseModel):
    model: str = "qwen2.5"
    base_url: str = "http://localhost:11434"


class EmbeddingConfig(BaseModel):
    provider: str = "dashscope"
    model: str = "text-embedding-v1"
    dimension: int = 1536
    api_key: str = ""
    base_url: str = ""


class SearchConfig(BaseModel):
    max_results: int = 5


class AgentConfig(BaseModel):
    model_call_limit: int = 8
    tool_call_limit: int = 6
    session_model_call_limit: int = 40
    session_tool_call_limit: int = 30
    parallel_tool_execution: bool = True
    max_parallel_tools: int = 4

    model_config = {"protected_namespaces": ()}


class MemoryConfig(BaseModel):
    strategy: str = "summary"
    sliding_window_size: int = 10
    summary_max_turns: int = 6
    recent_turns: int = 4
    recent_max_chars: int = 2200
    summary_max_chars: int = 1400


class RetrievalConfig(BaseModel):
    child_chunk_size: int = 500
    parent_chunk_size: int = 2000
    vector_top_k: int = 10
    keyword_top_k: int = 10
    rrf_k: int = 60
    min_vector_similarity: float = 0.08
    min_keyword_score: float = 0.05
    max_parent_chunks: int = 5
    evidence_budget_per_child: int = 1500
    evidence_budget_total: int = 4000
    enable_semantic_compress: bool = True


class DocumentConfig(BaseModel):
    chunk_strategies: List[str] = ["structural", "recursive", "semantic", "llm"]
    max_child_chunks: int = 500
    default_strategy: str = "structural,recursive"


class MCPConfig(BaseModel):
    servers: List[Dict] = Field(default_factory=list)


class IntentConfig(BaseModel):
    """意图判断关键词配置"""
    capability_hints: List[str] = ["你都能干什么", "你能做什么", "你可以做什么", "你会什么", "你是谁", "怎么用你", "能帮我什么"]
    chitchat_hints: List[str] = ["你好", "您好", "hello", "hi", "谢谢", "感谢", "再见", "拜拜", "在吗", "干嘛", "嗨"]
    tool_needed_hints: List[str] = ["搜索", "查询", "最新", "现在", "今日", "天气", "新闻", "股价", "汇率", "实时", "今天几", "现在几", "几点", "几号", "星期几", "时间", "日期"]
    open_chat_hints: List[str] = ["天气", "温度", "下雨", "新闻", "股价", "汇率", "热搜", "今天", "明天", "最新", "现在"]
    simple_chitchat: List[str] = ["1", "。", "", "啊", "嗯", "哦", "呃"]
    greeting_hints: List[str] = ["你好", "您好", "hi", "hello", "嗨", "嘿"]


class RerankConfig(BaseModel):
    provider: str = "siliconflow"
    api_key: str = ""
    model: str = "BAAI/bge-reranker-v2-m3"
    min_score: float = 0.05  # 最低相关分阈值，低于此分的结果会被过滤


# ── 顶层 Settings ───────────────────────────────────────

class Settings(BaseSettings):
    """应用配置 — 唯一入口"""

    # 服务
    server_host: str = "0.0.0.0"
    server_port: int = 8001
    server_reload: bool = False

    # 安全
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_hours: int = 24
    cors_allowed_origins: str = "http://localhost:5173"

    # 子配置
    mysql: MySQLConfig = Field(default_factory=MySQLConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @classmethod
    def load(cls) -> "Settings":
        """从 YAML + .env 统一加载配置"""
        from app.utils.env import load_dotenv

        load_dotenv()
        yaml_config = load_yaml_config()

        if not yaml_config:
            return cls()

        # 解析服务配置
        server_cfg = yaml_config.get("server", {})

        # 解析 MySQL
        mysql_cfg = yaml_config.get("database", {}).get("mysql", {})
        mysql = MySQLConfig(
            host=resolve_env(mysql_cfg.get("host", "localhost")),
            port=mysql_cfg.get("port", 3306),
            username=mysql_cfg.get("username", "root"),
            password=resolve_env(mysql_cfg.get("password", "password")),
            database=mysql_cfg.get("database", "super"),
        )

        # 解析 LLM
        llm_provider = yaml_config.get("llm", {}).get("provider", "dashscope")
        llm_cfg = yaml_config.get("llm", {}).get(llm_provider, {})
        llm = LLMConfig(
            provider=llm_provider,
            api_key=resolve_env(llm_cfg.get("api_key", "")),
            model=llm_cfg.get("model", "qwen-max"),
            max_tokens=llm_cfg.get("max_tokens", 4096),
            temperature=llm_cfg.get("temperature", 0.7),
            base_url=llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/api/v1"),
        )

        # 解析 Ollama
        ollama_cfg = yaml_config.get("llm", {}).get("ollama", {})
        ollama = OllamaConfig(
            model=ollama_cfg.get("model", "qwen2.5"),
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        )

        # 解析 Embedding
        emb_provider = yaml_config.get("embedding", {}).get("provider", "dashscope")
        emb_cfg = yaml_config.get("embedding", {}).get(emb_provider, {})
        embedding = EmbeddingConfig(
            provider=emb_provider,
            model=emb_cfg.get("model", "text-embedding-v1"),
            dimension=emb_cfg.get("dimension", 1536),
            api_key=resolve_env(emb_cfg.get("api_key", "")),
            base_url=emb_cfg.get("base_url", ""),
        )

        # 解析 Neo4j
        neo4j_cfg = yaml_config.get("graph_db", {}).get("neo4j", {})
        neo4j = Neo4jConfig(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            username=neo4j_cfg.get("username", "neo4j"),
            password=resolve_env(neo4j_cfg.get("password", "password")),
        )

        # 解析 Elasticsearch
        es_cfg = yaml_config.get("keyword_search", {}).get("elasticsearch", {})
        es_hosts_raw = es_cfg.get("hosts", ["http://localhost:9200"])
        es_hosts = [resolve_env(h) if isinstance(h, str) else h for h in es_hosts_raw]
        elasticsearch = ElasticsearchConfig(
            hosts=es_hosts,
            index=es_cfg.get("index", "super_agent_keywords"),
        )

        # 解析 MinIO
        minio_cfg = yaml_config.get("object_storage", {}).get("minio", {})
        minio = MinioConfig(
            endpoint=resolve_env(minio_cfg.get("endpoint", "localhost:9000")),
            access_key=resolve_env(minio_cfg.get("access_key", "minioadmin")),
            secret_key=resolve_env(minio_cfg.get("secret_key", "minioadmin")),
            bucket=minio_cfg.get("bucket", "super-agent"),
            secure=minio_cfg.get("secure", False),
        )

        # 解析 Kafka
        kafka_cfg = yaml_config.get("message_queue", {}).get("kafka", {})
        kafka = KafkaConfig(
            bootstrap_servers=resolve_env(kafka_cfg.get("bootstrap_servers", "localhost:9092")),
            topic=kafka_cfg.get("topic", "super-agent-doc-processing"),
        )

        # 解析 Redis
        redis_cfg = yaml_config.get("redis", {})
        redis_pwd = redis_cfg.get("password", "")
        redis = RedisConfig(
            host=resolve_env(redis_cfg.get("host", "localhost")),
            port=redis_cfg.get("port", 6379),
            db=redis_cfg.get("db", 0),
            password=redis_pwd if redis_pwd else None,
            prefix=redis_cfg.get("prefix", "super_agent:"),
        )

        # 解析 Rerank
        rerank_cfg = yaml_config.get("rerank", {})
        rerank_provider = rerank_cfg.get("provider", "siliconflow")
        si_cfg = rerank_cfg.get(rerank_provider, {})
        rerank = RerankConfig(
            provider=rerank_provider,
            api_key=resolve_env(si_cfg.get("api_key", "")),
            model=si_cfg.get("model", "BAAI/bge-reranker-v2-m3"),
            min_score=si_cfg.get("min_score", 0.3),
        )

        # 解析内存配置
        mem_cfg = yaml_config.get("memory", {})
        memory = MemoryConfig(**mem_cfg) if mem_cfg else MemoryConfig()

        # 解析 Agent 配置
        agent_cfg = yaml_config.get("agent", {})
        agent = AgentConfig(**agent_cfg) if agent_cfg else AgentConfig()

        # 解析检索配置
        ret_cfg = yaml_config.get("retrieval", {})
        retrieval = RetrievalConfig(**ret_cfg) if ret_cfg else RetrievalConfig()

        # 解析文档配置
        doc_cfg = yaml_config.get("document", {})
        document = DocumentConfig(
            chunk_strategies=doc_cfg.get("chunk_strategies", ["structural", "recursive", "semantic", "llm"]),
            max_child_chunks=doc_cfg.get("max_child_chunks", 500),
            default_strategy=doc_cfg.get("default_strategy", "structural,recursive"),
        ) if doc_cfg else DocumentConfig()

        # 解析搜索配置
        search_cfg = yaml_config.get("search", {}).get("duckduckgo", {})
        search = SearchConfig(max_results=search_cfg.get("max_results", 5))

        # 解析 MCP
        mcp_list = yaml_config.get("mcp", {}).get("servers", [])
        mcp = MCPConfig(servers=mcp_list)

        return cls(
            server_host=server_cfg.get("host", "0.0.0.0"),
            server_port=server_cfg.get("port", 8001),
            server_reload=server_cfg.get("reload", False),
            mysql=mysql,
            chroma=ChromaConfig(**yaml_config.get("vector_store", {}).get("chroma", {})),
            elasticsearch=elasticsearch,
            neo4j=neo4j,
            minio=minio,
            kafka=kafka,
            redis=redis,
            llm=llm,
            ollama=ollama,
            embedding=embedding,
            search=search,
            agent=agent,
            memory=memory,
            retrieval=retrieval,
            document=document,
            rerank=rerank,
            mcp=mcp,
            intent=IntentConfig(),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    """强制重新加载配置"""
    global _settings
    _settings = Settings.load()
    return _settings
