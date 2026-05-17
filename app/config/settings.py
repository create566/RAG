"""
配置管理模块
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MySQLConfig(BaseModel):
    host: str = "localhost"
    port: int = 3306
    username: str = "root"
    password: str = "password"
    database: str = "super_agent"

    @property
    def url(self) -> str:
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


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


class DashScopeLLMConfig(BaseModel):
    api_key: str = ""
    model: str = "qwen-turbo"
    max_tokens: int = 4096
    temperature: float = 0.7
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"


class EmbeddingConfig(BaseModel):
    provider: str = "dashscope"
    model: str = "text-embedding-ada-002"
    dimension: int = 1536


class TavilyConfig(BaseModel):
    api_key: str = ""
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
    min_vector_similarity: float = 0.5
    min_keyword_score: float = 0.3
    max_parent_chunks: int = 5
    evidence_budget_per_child: int = 1500
    evidence_budget_total: int = 4000


class DocumentConfig(BaseModel):
    chunk_strategies: List[str] = ["structural", "recursive", "semantic", "llm"]
    max_child_chunks: int = 500
    default_strategy: str = "structural"


class Settings(BaseSettings):
    """应用配置"""
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    server_reload: bool = True

    mysql: MySQLConfig = Field(default_factory=MySQLConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    llm: DashScopeLLMConfig = Field(default_factory=DashScopeLLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)

    @classmethod
    def load_from_yaml(cls, path: str) -> "Settings":
        """从YAML文件加载配置"""
        # 确保加载 .env 文件中的环境变量
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent / ".env")

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 处理环境变量
        def resolve_env(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_key = value[2:-1]
                return os.environ.get(env_key, "")
            return value

        server_cfg = config_dict.get("server", {})
        db_cfg = config_dict.get("database", {}).get("mysql", {})
        llm_cfg = config_dict.get("llm", {}).get("dashscope", {})

        return cls(
            server_host=server_cfg.get("host", "0.0.0.0"),
            server_port=server_cfg.get("port", 8000),
            server_reload=server_cfg.get("reload", True),
            mysql=MySQLConfig(**db_cfg),
            chroma=ChromaConfig(**config_dict.get("vector_store", {}).get("chroma", {})),
            elasticsearch=ElasticsearchConfig(**config_dict.get("keyword_search", {}).get("elasticsearch", {})),
            neo4j=Neo4jConfig(**config_dict.get("graph_db", {}).get("neo4j", {})),
            minio=MinioConfig(**config_dict.get("object_storage", {}).get("minio", {})),
            kafka=KafkaConfig(**config_dict.get("message_queue", {}).get("kafka", {})),
            llm=DashScopeLLMConfig(
                api_key=resolve_env(llm_cfg.get("api_key", "")),
                model=llm_cfg.get("model", "qwen-turbo"),
                max_tokens=llm_cfg.get("max_tokens", 4096),
                temperature=llm_cfg.get("temperature", 0.7)
            ),
            embedding=EmbeddingConfig(**config_dict.get("embedding", {}).get("dashscope", {})),
            tavily=TavilyConfig(**config_dict.get("search", {}).get("tavily", {})),
            agent=AgentConfig(**config_dict.get("agent", {})),
            memory=MemoryConfig(**config_dict.get("memory", {})),
            retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
            document=DocumentConfig(**config_dict.get("document", {}))
        )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取配置单例"""
    global _settings
    if _settings is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if config_path.exists():
            _settings = Settings.load_from_yaml(str(config_path))
        else:
            _settings = Settings()
    return _settings