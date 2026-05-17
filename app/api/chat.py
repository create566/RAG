"""
聊天API接口 - 对标Java的BusinessChatController
"""
from typing import Optional
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
import json

from app.models.chat import ChatRequest, ChatResponse, SourceReference
from app.rag import ChatPreparationOrchestrator, ConversationExecutionPlan, ExecutionMode as RagExecutionMode
from app.agent import ClarificationExecutor, AgentExecutor, ReActAgent, ExecutionMode
from app.agent.skill import SkillManager, get_skill_manager
from app.agent.mcp import get_mcp_provider
from app.memory import ConversationMemoryService, SummaryCompressionMemoryStrategy, MySQLMemoryStrategy
from app.core import create_llm_service, RagPromptAssemblyService
from app.core.rerank_service import SiliconFlowRerankService, MockRerankService
from app.retrieval import RagRetrievalContext, RagRetrievalEngine, VectorRetrievalChannel, KeywordRetrievalChannel
from app.knowledge import KnowledgeRouteService, DocumentQuestionRouter, DocumentService
from app.core.graph_engine import create_graph_engine

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatService:
    """聊天服务 - 整合所有组件"""

    def __init__(self):
        self.config = self._load_config()

        # 解析环境变量
        def resolve_env(value):
            if isinstance(value, str):
                # 处理 ${VAR:-default} 格式
                if value.startswith("${") and ":-" in value and value.endswith("}"):
                    inner = value[2:-1]
                    var_name, default = inner.split(":-", 1)
                    return os.environ.get(var_name, default)
                # 处理 ${VAR} 格式
                elif value.startswith("${") and value.endswith("}"):
                    return os.environ.get(value[2:-1], "")
            return value

        # 初始化LLM服务
        llm_config = self.config.get("llm", {})
        dashscope_config = llm_config.get("dashscope", {})
        dashscope_config["api_key"] = resolve_env(dashscope_config.get("api_key", ""))

        self.llm_service = create_llm_service(
            provider=llm_config.get("provider", "dashscope"),
            config={**llm_config, "dashscope": dashscope_config}
        )

        # 初始化记忆服务
        memory_config = self.config.get("memory", {})

        # 获取 MySQL 配置
        mysql_config = self.config.get("database", {}).get("mysql", {})
        db_host = resolve_env(mysql_config.get("host", "localhost"))
        db_port = mysql_config.get("port", 3306)
        db_user = mysql_config.get("username", "root")
        db_password = resolve_env(mysql_config.get("password", "123456"))
        db_database = mysql_config.get("database", "super")

        memory_strategy = MySQLMemoryStrategy(
            host=db_host,
            port=db_port,
            username=db_user,
            password=db_password,
            database=db_database,
            recent_turns=memory_config.get("recent_turns", 4),
            recent_max_chars=memory_config.get("recent_max_chars", 2200),
            summary_max_turns=memory_config.get("summary_max_turns", 6),
            summary_max_chars=memory_config.get("summary_max_chars", 1400),
            llm_service=self.llm_service
        )
        self.memory_service = ConversationMemoryService(memory_strategy)

        # 初始化向量存储
        vector_config = self.config.get("vector_store", {})
        if vector_config.get("chroma"):
            vector_config = {**vector_config, **vector_config.get("chroma", {})}
        from app.core.chroma_client import create_vector_store
        self.vector_store = create_vector_store(vector_config)

        # 初始化RAG检索引擎（完整pipeline）
        retrieval_config = self.config.get("retrieval", {})

        # 初始化关键词检索通道（Elasticsearch + BM25）
        from elasticsearch import Elasticsearch
        es_config = self.config.get("keyword_search", {}).get("elasticsearch", {})
        es_hosts_raw = es_config.get("hosts", ["http://localhost:9200"])
        # 解析环境变量
        es_hosts = [resolve_env(h) if isinstance(h, str) else h for h in es_hosts_raw]
        self.es_client = Elasticsearch(hosts=es_hosts)

        # 使用 pipeline.py 里的 ES 版 KeywordRetrievalChannel
        from app.retrieval.pipeline import KeywordRetrievalChannel as ESKeywordRetrievalChannel
        self.keyword_channel = ESKeywordRetrievalChannel(
            elasticsearch_client=self.es_client,
            config={
                "index": es_config.get("index", "super_agent_keywords"),
                "min_keyword_score": retrieval_config.get("min_keyword_score", 0.3)
            }
        )

        # 初始化Rerank服务（SiliconFlow）
        rerank_config = self.config.get("rerank", {})
        rerank_api_key = resolve_env(rerank_config.get("siliconflow", {}).get("api_key", ""))
        if rerank_api_key:
            self.rerank_service = SiliconFlowRerankService(
                api_key=rerank_api_key,
                model=rerank_config.get("siliconflow", {}).get("model", "BAAI/bge-reranker-base")
            )
        else:
            self.rerank_service = None

        # 使用已索引的keyword_channel（复用上面的实例）
        self.rag_engine = RagRetrievalEngine(
            vector_channel=VectorRetrievalChannel(
                embedding_service=self.llm_service,
                vector_store=self.vector_store,
                config=retrieval_config
            ),
            keyword_channel=self.keyword_channel,  # 复用已索引的实例
            rerank_service=self.rerank_service,
            llm_service=self.llm_service,
            config={
                "rrf_k": retrieval_config.get("rrf_k", 60),
                "max_parent_chunks": retrieval_config.get("max_parent_chunks", 5),
                "evidence_budget_per_child": retrieval_config.get("evidence_budget_per_child", 1500),
                "evidence_budget_total": retrieval_config.get("evidence_budget_total", 4000),
                "enable_semantic_compress": retrieval_config.get("enable_semantic_compress", False),
                "vector_top_k": retrieval_config.get("vector_top_k", 10),
                "keyword_top_k": retrieval_config.get("keyword_top_k", 10)
            }
        )

        # 初始化图查询引擎（Neo4j）
        graph_config = self.config.get("graph_db", {})
        self.graph_query_engine = None
        if graph_config.get("neo4j", {}).get("uri"):
            try:
                from app.core.graph_engine import create_graph_engine
                self.graph_query_engine = create_graph_engine(
                    provider="neo4j",
                    config={
                        "neo4j": {
                            "uri": graph_config["neo4j"]["uri"],
                            "username": graph_config["neo4j"].get("username", "neo4j"),
                            "password": graph_config["neo4j"].get("password", "")
                        }
                    }
                )
                print(f"[GRAPH] 图查询引擎初始化成功")
            except Exception as e:
                print(f"[GRAPH] 图查询引擎初始化失败: {e}")
                self.graph_query_engine = None

        # 初始化Prompt组装服务
        self.prompt_assembler = RagPromptAssemblyService(config={
            "per_sub_question_budget": retrieval_config.get("per_sub_question_budget", 1500),
            "total_budget": retrieval_config.get("total_budget", 4000)
        })

        # 初始化文档服务
        self.document_service = DocumentService(vector_store=self.vector_store)

        # 初始化知识路由服务
        self.knowledge_route_service = KnowledgeRouteService(self.llm_service, self.document_service)

        # 初始化文档问题路由
        self.document_router = DocumentQuestionRouter(
            graph_query_engine=self.graph_query_engine
        )

        # 初始化编排器
        self.orchestrator = ChatPreparationOrchestrator(
            llm_service=self.llm_service,
            memory_service=self.memory_service,
            document_router=self.document_router,
            knowledge_route_service=self.knowledge_route_service,
            document_service=self.document_service,
            config=self.config.get("retrieval", {})
        )

        # 初始化ReAct Agent
        tools = []
        # 使用 DuckDuckGo 搜索（免费无需API Key）
        try:
            from app.agent.react import DuckDuckGoSearchTool
            search_config = self.config.get("search", {}).get("duckduckgo", {})
            max_results = search_config.get("max_results", 5) if search_config else 5
            ddg_tool = DuckDuckGoSearchTool(max_results=max_results)
            tools.append(ddg_tool)
            # 注册 web_search 工具到技能管理器
            skill_manager = get_skill_manager()
            skill_manager.register_tool("web_search", ddg_tool)
            print(f"[SEARCH] DuckDuckGo search tool registered, max_results={max_results}")
        except ImportError as e:
            print(f"[SEARCH] DuckDuckGo not available: {e}")

        # 获取当前时间工具
        from app.agent.react import GetCurrentTimeTool
        time_tool = GetCurrentTimeTool()
        tools.append(time_tool)
        skill_manager = get_skill_manager()
        skill_manager.register_tool("get_current_time", time_tool)
        print(f"[TOOL] GetCurrentTime tool registered")

        # 加载 Skills 配置
        skills_config = self.config.get("skills", [])
        skill_manager = get_skill_manager()
        skill_manager.load_skills_from_config(skills_config)
        print(f"[SKILL] Loaded {len(skill_manager.list_skills())} skills")

        agent_config = self.config.get("agent", {})
        self.react_agent = ReActAgent(self.llm_service, tools, agent_config)

    def _index_keyword_channel(self):
        """索引关键词检索通道"""
        try:
            if not self.vector_store:
                print("[KEYWORD] 向量存储为空")
                return
            # 从向量库获取所有文档
            results = self.vector_store.peek(limit=1000)
            print(f"[KEYWORD] peek结果: keys={results.keys() if results else None}")
            if not results or not results.get("documents"):
                print("[KEYWORD] peek返回空")
                return

            # documents是字符串列表，每个元素是一个文档块
            all_docs = results.get("documents", [])
            all_metas = results.get("metadatas", [])

            print(f"[KEYWORD] 文档数: {len(all_docs)}, metadata数: {len(all_metas)}")

            chunks = []
            for i, doc in enumerate(all_docs):
                if i >= len(all_metas):
                    continue
                meta = all_metas[i] if isinstance(all_metas[i], dict) else {}
                chunks.append({
                    "content": doc,
                    "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
                    "document_id": meta.get("document_id", ""),
                    "document_name": meta.get("document_name", ""),
                    "metadata": meta
                })

            print(f"[KEYWORD] 构建chunks: {len(chunks)} 个")
            if chunks:
                self.keyword_channel.index_documents(chunks)
                print(f"[INFO] 关键词索引完成，共 {len(chunks)} 个文档块")
        except Exception as e:
            import traceback
            print(f"[WARN] 关键词索引失败: {e}")
            traceback.print_exc()

    def _load_config(self) -> dict:
        """加载配置"""
        try:
            import yaml
            from pathlib import Path
            from dotenv import load_dotenv

            # 加载 .env 文件
            load_dotenv(Path(__file__).parent.parent.parent / ".env")

            config_path = Path(__file__).parent.parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except:
            pass
        return {}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        import sys
        print(f"[CHAT API] START request: question={request.question}, chat_mode={request.chat_mode}", flush=True)
        try:
            conversation_id = request.conversation_id or f"conv_{id(request)}"

            # 准备任务信息
            task_info = {
                "conversation_id": conversation_id,
                "question": request.question,
                "chat_mode": request.chat_mode.value,
                "selected_document_id": request.selected_document_id,
                "selected_document_name": request.selected_document_name,
                "selected_task_id": request.selected_task_id,
                "current_date": None,
                "current_date_text": ""
            }

            # 执行编排
            plan = await self.orchestrator.prepare(task_info)
            print(f"[DEBUG] Plan mode: {plan.mode}, value: {plan.mode.value}")

            # 根据执行模式调用不同的执行器
            if plan.mode == RagExecutionMode.CLARIFICATION:
                executor = ClarificationExecutor(self.llm_service)
                result = await executor.execute({
                    "question": plan.original_question,
                    "clarification_reply": plan.clarification_reply,
                    "clarification_options": plan.clarification_options
                })
            elif plan.mode == RagExecutionMode.REACT_AGENT:
                executor = AgentExecutor(self.react_agent)
                # 加载记忆上下文，转换为 Agent 期望的格式
                memory_context = await self.memory_service.load_memory_context(conversation_id, request.user_id)
                history_text = memory_context.get("recent_transcript", "") or ""
                # 将字符串格式转为 List[Dict] 格式
                history = []
                if history_text:
                    for line in history_text.split("\n"):
                        if line.startswith("用户:"):
                            history.append({"role": "user", "content": line[3:].strip()})
                        elif line.startswith("助手:"):
                            history.append({"role": "assistant", "content": line[3:].strip()})
                result = await executor.execute({
                    "question": plan.original_question,
                    "history": history
                })
            else:
                # RAG检索模式 - 使用完整pipeline
                try:
                    retrieval_context = None  # 初始化，避免 UnboundLocalError

                    # 检查是否是图查询优先模式
                    if plan.navigation_decision and plan.navigation_decision.execution_mode == RagExecutionMode.GRAPH_THEN_EVIDENCE:
                        # 图查询模式：直接从 Neo4j 获取章节内容
                        graph_data = plan.navigation_decision.structure_anchor.get("graph_data", {}) if plan.navigation_decision.structure_anchor else {}

                        if graph_data.get("status") == "success":
                            graph_type = graph_data.get("type", "")
                            if graph_type == "structure":
                                chapters = graph_data.get("chapters", [])
                                answer = f"《{graph_data.get('document_title', '')}》包含以下章节：\n"
                                for ch in chapters:
                                    answer += f"- {ch}\n"
                            elif graph_type == "section":
                                answer = f"**{graph_data.get('chapter_title', '')}**\n\n{graph_data.get('chapter_content', '')}"
                            elif graph_type == "item":
                                answer = f"**{graph_data.get('item_title', '')}**\n\n{graph_data.get('item_content', '')}"
                            else:
                                # fallback to retrieval
                                retrieval_context = await self.rag_engine.retrieve(plan, user_id=request.user_id)
                                if retrieval_context.is_empty():
                                    print(f"[RAG] Unknown graph type + no evidence, fallback to LLM chat")
                                    answer = await self.llm_service.chat(
                                        plan.original_question,
                                        system_prompt="你是一个智能助手，请直接回答用户问题。"
                                    )
                                else:
                                    prompt_dict = self.prompt_assembler.assemble(plan, retrieval_context)
                                    answer = await self.llm_service.chat(
                                        prompt_dict["user_prompt"],
                                        system_prompt=prompt_dict["system_prompt"]
                                    )
                        else:
                            # 图查询失败，fallback到检索
                            retrieval_context = await self.rag_engine.retrieve(plan, user_id=request.user_id)
                            if retrieval_context.is_empty():
                                print(f"[RAG] Graph failed + no evidence, fallback to LLM chat")
                                answer = await self.llm_service.chat(
                                    plan.original_question,
                                    system_prompt="你是一个智能助手，请直接回答用户问题。"
                                )
                            else:
                                prompt_dict = self.prompt_assembler.assemble(plan, retrieval_context)
                                answer = await self.llm_service.chat(
                                    prompt_dict["user_prompt"],
                                    system_prompt=prompt_dict["system_prompt"]
                                )
                    else:
                        # 普通 RAG 检索模式
                        retrieval_context = await self.rag_engine.retrieve(plan, user_id=request.user_id)

                        if retrieval_context.is_empty():
                            # 检索无结果 → 降级为 LLM 自由回答
                            print(f"[RAG] No evidence found, fallback to LLM chat")
                            answer = await self.llm_service.chat(
                                plan.original_question,
                                system_prompt="你是一个智能助手。知识库中没有找到相关信息，请基于你的知识直接回答用户问题。"
                            )
                        else:
                            prompt_dict = self.prompt_assembler.assemble(plan, retrieval_context)
                            print(f"[LLM] Calling with prompt length: user={len(prompt_dict['user_prompt'])}, system={len(prompt_dict['system_prompt'])}")
                            answer = await self.llm_service.chat(
                                prompt_dict["user_prompt"],
                                system_prompt=prompt_dict["system_prompt"]
                            )
                            print(f"[LLM] Received answer: {repr(answer)[:200]}")

                    # 获取来源（图查询模式可能没有 retrieval_context）
                    sources = []
                    if retrieval_context:
                        for ref in retrieval_context.flatten_references()[:5]:
                            sources.append({
                                "reference_id": ref.get("reference_id", ""),
                                "document_name": ref.get("document_name", ""),
                                "section_path": ref.get("section_path", ""),
                                "channel": ref.get("channel", "")
                            })

                    # 生成建议问题
                    suggested_questions = []

                    class Result:
                        def __init__(self, answer, sources, suggested_questions):
                            self.mode = ExecutionMode.RETRIEVAL
                            self.success = True
                            self.content = answer
                            self.sources = sources
                            self.suggested_questions = suggested_questions

                    result = Result(answer, sources, suggested_questions)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    answer = f"检索失败: {str(e)}"

                    class Result:
                        def __init__(self):
                            self.mode = ExecutionMode.RETRIEVAL
                            self.success = False
                            self.content = answer
                            self.sources = []
                            self.suggested_questions = []
                    result = Result()

            # 保存对话记忆
            try:
                content = result.content if hasattr(result, 'content') else str(result)
                await chat_service.memory_service.save_exchange(conversation_id, request.user_id, request.question, content)

                # 同时保存 conversation 记录（用于侧边栏列表）
                import pymysql
                conv_conn = pymysql.connect(
                    host="localhost", port=3306, user="root", password="123456",
                    database="super", charset="utf8mb4"
                )
                conv_cur = conv_conn.cursor()
                # 确保表存在
                conv_cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation (
                        id VARCHAR(64) PRIMARY KEY,
                        user_id INT NOT NULL DEFAULT 0,
                        title VARCHAR(255) DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        summary TEXT,
                        INDEX idx_user_id (user_id),
                        INDEX idx_updated_at (updated_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                conv_conn.commit()
                # 确保 message 表存在
                conv_cur.execute("""
                    CREATE TABLE IF NOT EXISTS message (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        conversation_id VARCHAR(64) NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_conversation_id (conversation_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                conv_conn.commit()
                # 保存会话记录
                conv_cur.execute(
                    "INSERT INTO conversation (id, user_id, title, summary) VALUES (%s, %s, %s, '') "
                    "ON DUPLICATE KEY UPDATE updated_at = NOW(), user_id = COALESCE(user_id, VALUES(user_id))",
                    (conversation_id, request.user_id or 0, request.question[:100])
                )
                # 保存用户消息和助手回复到 message 表
                conv_cur.execute(
                    "INSERT INTO message (conversation_id, role, content) VALUES (%s, 'user', %s)",
                    (conversation_id, request.question)
                )
                conv_cur.execute(
                    "INSERT INTO message (conversation_id, role, content) VALUES (%s, 'assistant', %s)",
                    (conversation_id, content)
                )
                conv_conn.commit()
                print(f"[CONV SAVE] id={conversation_id[:12]}, user_id={request.user_id}, title={request.question[:40]}, messages=2")
                conv_conn.close()
            except Exception as mem_err:
                print(f"保存记忆失败: {mem_err}")

            # 构建响应
            print(f"[DEBUG] Building response, result type: {type(result)}")
            sources = []
            try:
                result_sources = result.sources if hasattr(result, 'sources') and result.sources else []
                print(f"[DEBUG] result_sources: {type(result_sources)}, len: {len(result_sources)}")
                for i, src in enumerate(result_sources[:5]):
                    print(f"[DEBUG] src[{i}]: type={type(src)}, value={repr(src)[:100]}")
                    if src is None:
                        print(f"[WARN] Skipping None source at {i}")
                        continue
                    if not isinstance(src, dict):
                        print(f"[WARN] Skipping non-dict source at {i}: {type(src)}")
                        continue
                    src_ref = SourceReference(
                        reference_id=str(src.get("reference_id", "") or ""),
                        document_name=str(src.get("document_name", "") or ""),
                        section_path=str(src.get("section_path", "") or ""),
                        channel=str(src.get("channel", "") or "")
                    )
                    sources.append(src_ref)
                    print(f"[DEBUG] Added source: {src.get('document_name')}/{src.get('section_path')}")
            except Exception as src_err:
                print(f"构建来源失败: {src_err}")

            # 获取suggested_questions
            suggested_questions = []
            try:
                if hasattr(result, 'suggested_questions') and result.suggested_questions:
                    suggested_questions = list(result.suggested_questions)
            except Exception as e:
                print(f"获取建议问题失败: {e}")

            return ChatResponse(
                answer=str(result.content) if hasattr(result, 'content') else "无内容",
                sources=sources,
                suggested_questions=suggested_questions,
                conversation_id=conversation_id
            )

        except Exception as e:
            import traceback
            err_msg = f"处理出错: {str(e)}\n{traceback.format_exc()}"
            print(err_msg, flush=True)
            return ChatResponse(
                answer=err_msg,
                sources=[],
                suggested_questions=[],
                conversation_id=request.conversation_id or "error"
            )


# 全局聊天服务实例
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    return await chat_service.chat(request)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口"""
    async def generate():
        try:
            try:
                response = await chat_service.chat(request)
            except Exception as inner_e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'answer': f'调用chat失败: {str(inner_e)}', 'done': True}, ensure_ascii=False)}\n\n"
                return

            if response is None:
                yield f"data: {json.dumps({'answer': '响应为None', 'done': True}, ensure_ascii=False)}\n\n"
            else:
                try:
                    answer_text = response.answer if hasattr(response, 'answer') and response.answer else '无响应'
                    conv_id = response.conversation_id if hasattr(response, 'conversation_id') and response.conversation_id else None
                    print(f"[STREAM] answer_text length: {len(answer_text)}, answer repr: {repr(answer_text)[:100]}")
                    yield f"data: {json.dumps({'answer': answer_text, 'conversation_id': conv_id, 'done': True}, ensure_ascii=False)}\n\n"
                except Exception as resp_e:
                    yield f"data: {json.dumps({'answer': f'响应处理失败: {str(resp_e)}', 'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            error_msg = f"错误: {str(e)}\n{traceback.format_exc()}"
            yield f"data: {json.dumps({'answer': error_msg, 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """获取会话历史"""
    memory_context = await chat_service.memory_service.load_memory_context(conversation_id)
    return {
        "conversation_id": conversation_id,
        "context": memory_context
    }


@router.get("/conversations")
async def list_conversations(user_id: int = None):
    """
    列出用户的会话列表，按日期分组
    返回格式: { date_groups: [{ date: "今天"/"昨天"/"2026-05-15", conversations: [...] }] }
    """
    import pymysql
    from datetime import datetime, timedelta

    if not user_id:
        return {"success": False, "error": "需要 user_id 参数"}

    try:
        conn = pymysql.connect(
            host="localhost", port=3306, user="root", password="123456",
            database="super", charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # 自动建表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id VARCHAR(64) PRIMARY KEY,
                user_id INT NOT NULL DEFAULT 0,
                title VARCHAR(255) DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                summary TEXT,
                INDEX idx_user_id (user_id),
                INDEX idx_updated_at (updated_at),
                INDEX idx_user_updated (user_id, updated_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS message (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                conversation_id VARCHAR(64) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_conversation_id (conversation_id),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()

        # 先查出总数（不限 user_id）
        cursor.execute("SELECT COUNT(*) as cnt FROM conversation")
        total = cursor.fetchone()["cnt"]
        print(f"[CONV LIST] total conversations in DB: {total}")

        cursor.execute(
            "SELECT id, user_id, title, summary, created_at, updated_at "
            "FROM conversation WHERE user_id = %s "
            "ORDER BY updated_at DESC LIMIT 50",
            (user_id,)
        )
        convs = cursor.fetchall()
        print(f"[CONV LIST] user_id={user_id} matched: {len(convs)}, ids: {[c['id'][:12] for c in convs[:5]]}")

        # 获取每个会话的第一条用户消息作为标题
        for conv in convs:
            if not conv.get("title"):
                cursor.execute(
                    "SELECT content FROM message WHERE conversation_id = %s AND role = 'user' ORDER BY id ASC LIMIT 1",
                    (conv["id"],)
                )
                first_msg = cursor.fetchone()
                if first_msg:
                    conv["title"] = first_msg["content"][:50]

        conn.close()

        # 按日期分组
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        date_groups = {}

        for conv in convs:
            conv_date = conv["updated_at"].date() if conv.get("updated_at") else today
            if conv_date == today:
                date_label = "今天"
            elif conv_date == yesterday:
                date_label = "昨天"
            else:
                date_label = conv_date.strftime("%Y-%m-%d")

            if date_label not in date_groups:
                date_groups[date_label] = []
            date_groups[date_label].append({
                "id": conv["id"],
                "title": conv.get("title") or conv.get("summary") or "新对话",
                "created_at": conv["created_at"].isoformat() if conv.get("created_at") else None,
                "updated_at": conv["updated_at"].isoformat() if conv.get("updated_at") else None,
            })

        result = [{"date": k, "conversations": v} for k, v in date_groups.items()]

        return {"success": True, "date_groups": result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """获取会话的全部消息"""
    import pymysql

    try:
        conn = pymysql.connect(
            host="localhost", port=3306, user="root", password="123456",
            database="super", charset="utf8mb4"
        )
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(
            "SELECT id, role, content, created_at FROM message WHERE conversation_id = %s ORDER BY id ASC",
            (conversation_id,)
        )
        messages = [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "created_at": m["created_at"].isoformat() if m.get("created_at") else None
            }
            for m in cursor.fetchall()
        ]
        conn.close()
        return {"success": True, "conversation_id": conversation_id, "messages": messages}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除会话"""
    import pymysql
    try:
        conn = pymysql.connect(
            host="localhost", port=3306, user="root", password="123456",
            database="super", charset="utf8mb4"
        )
        cursor = conn.cursor()
        cursor.execute("DELETE FROM message WHERE conversation_id = %s", (conversation_id,))
        cursor.execute("DELETE FROM conversation WHERE id = %s", (conversation_id,))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/graph/document/{document_id}")
async def query_graph_document(document_id: int):
    """
    查询文档的图谱结构（测试用）
    通过 Apifox 测试 Neo4j 图谱查询功能
    """
    try:
        if not chat_service.graph_query_engine:
            return {"success": False, "error": "图查询引擎未初始化"}

        result = await chat_service.graph_query_engine.query_document_structure(document_id)
        return {
            "success": True,
            "document_id": document_id,
            "structure": result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/graph/document/{document_id}/chapter")
async def query_graph_chapter(document_id: int, section_hint: str = None):
    """
    查询文档特定章节内容（测试用）
    section_hint: 章节提示，如 "第3章"
    """
    try:
        if not chat_service.graph_query_engine:
            return {"success": False, "error": "图查询引擎未初始化"}

        if section_hint:
            result = await chat_service.graph_query_engine.query_section_content(document_id, section_hint)
        else:
            result = await chat_service.graph_query_engine.query_document_structure(document_id)

        return {
            "success": True,
            "document_id": document_id,
            "section_hint": section_hint,
            "result": result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/graph/cypher")
async def execute_cypher(cypher: str = None, query: str = None):
    """
    执行自定义 Cypher 查询（测试用）
    直接在 Neo4j 中执行 Cypher 语句
    """
    try:
        if not chat_service.graph_query_engine:
            return {"success": False, "error": "图查询引擎未初始化"}

        # query 参数优先（兼容不同客户端）
        cypher_query = query or cypher
        if not cypher_query:
            return {"success": False, "error": "请提供 cypher 或 query 参数"}

        from app.core.neo4j_client import Neo4jClient
        graph_config = chat_service.config.get("graph_db", {})
        neo4j_config = graph_config.get("neo4j", {})

        client = Neo4jClient(
            uri=neo4j_config.get("uri", "bolt://localhost:7687"),
            username=neo4j_config.get("username", "neo4j"),
            password=neo4j_config.get("password", "")
        )

        if not client.health_check():
            return {"success": False, "error": "Neo4j 连接失败"}

        results = client.query(cypher_query)
        return {
            "success": True,
            "cypher": cypher_query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}