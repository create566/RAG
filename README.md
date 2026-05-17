# Super Agent Python Backend

企业级AI智能体对话平台的Python实现，对标Java版Super Agent。

## 核心技术栈

| 技术 | 说明 |
|------|------|
| FastAPI | Web服务框架，对标Spring Boot |
| LangGraph | Agent状态机编排（ReAct执行器核心） |
| Chroma | 向量数据库（存储文档嵌入向量） |
| Elasticsearch | BM25关键词检索（ES索引） |
| Neo4j | 图数据库（文档章节图谱） |
| MySQL | 会话记忆持久化（摘要压缩策略） |
| Kafka | 异步消息队列 |
| MinIO | 对象存储 |
| DashScope | 通义千问大模型服务 |
| SiliconFlow | 重排服务（bge-reranker-v2-m3） |

## 主要功能

### RAG检索管道
- **双通道混合检索**：向量检索 + BM25关键词检索 + RRF融合
- **Parent-Child块聚合**：子块聚合到父块
- **证据预算控制**：单子问题1500字符，总量4000字符
- **SiliconFlow重排**：BAAI/bge-reranker-v2-m3模型

### 三层执行器
- **歧义追问执行器**：检测用户问题歧义，触发追问
- **RAG知识问答执行器**：完整RAG pipeline，意图匹配切换
- **ReAct Agent执行器**：基于LangGraph状态机，工具调用+Tavily搜索

### 文档处理
- **上传**：解析 → 切块 → 向量化 → 三存储（Chroma/ES/Neo4j）
- **Neo4j图谱构建**：Document节点 → Chapter关系 → Paragraph关系
- **ES BM25索引**：关键词检索，支持精确匹配

### 会话记忆
- **MySQL持久化**：summary_compression压缩策略
- **近期对话**：最近4轮完整记录
- **摘要压缩**：每6轮压缩一次，保留1400字符摘要

### 工具扩展
- **Skills声明式**：配置文件定义Agent技能
- **MCP协议**：动态工具发现和调用
- **Tavily搜索**：实时网络搜索工具

## 项目结构

```
super/
├── app/
│   ├── agent/           # ReAct执行器、Skills、MCP、工具
│   ├── api/             # API接口（chat、document）
│   ├── config/          # 配置管理（settings、yaml解析）
│   ├── core/            # 核心服务（LLM、向量、图谱、重排）
│   ├── documents/       # 文档处理（解析、切块）
│   ├── knowledge/       # 知识路由（文档服务、问题路由）
│   ├── memory/          # 会话记忆（MySQL策略）
│   ├── models/          # 数据模型
│   ├── rag/             # RAG编排（ChatOrchestrator）
│   └── retrieval/       # 检索管道（向量/关键词双通道、RRF）
├── config.yaml          # 配置文件
├── requirements.txt     # 依赖
└── run.py               # 启动脚本
```

## 配置文件说明 (config.yaml)

### 向量检索配置
```yaml
retrieval:
  min_vector_similarity: 0.08    # 向量相似度阈值
  min_keyword_score: 0.05        # BM25得分阈值
  rrf_k: 60                      # RRF融合参数
  vector_top_k: 10               # 向量检索返回数
  keyword_top_k: 10              # 关键词检索返回数
  evidence_budget_per_child: 1500
  evidence_budget_total: 4000
```

### Neo4j图谱配置
```yaml
graph_db:
  neo4j:
    uri: bolt://localhost:7687
    username: neo4j
    password: ${NEO4J_PASSWORD:-password}
```

## API接口

### 聊天接口
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/chat` | 聊天接口 |
| POST | `/api/chat/chat/stream` | 流式聊天 |
| GET | `/api/chat/conversation/{id}/history` | 会话历史 |

### Neo4j图谱测试接口
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/chat/graph/document/{document_id}` | 查询文档图谱结构 |
| GET | `/api/chat/graph/document/{document_id}/chapter?section_hint=第1章` | 查询章节内容 |
| GET | `/api/chat/graph/cypher?query=MATCH...` | 执行Cypher查询 |

### 文档管理接口
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/document/upload` | 上传文档（同步处理） |
| GET | `/api/document/list` | 列出已上传文档 |
| DELETE | `/api/document/{document_id}` | 删除文档 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量 (.env)
DASHSCOPE_API_KEY=your-api-key
NEO4J_PASSWORD=your-password
ELASTICSEARCH_HOSTS=http://localhost:9200

# 启动服务
python run.py
# 服务地址: http://localhost:8001
```

## 启动命令

```bash
# 查看端口占用
netstat -ano | findstr ":8001"

# 停止服务
taskkill /F /PID <PID>

# 启动服务
python run.py
#前端启动
npm run dev
```

## 验证服务

```bash
# 健康检查
curl http://localhost:8001/health

# 查看文档列表
curl http://localhost:8001/api/document/list

# 测试图谱查询
curl http://localhost:8001/api/chat/graph/document/<docume

sk-add2253bd4d246f0b9d41c2c74637536