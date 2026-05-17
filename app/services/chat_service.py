"""
聊天业务服务 — 从 api/chat.py 提取
"""
import json
import pymysql

from app.config import get_settings
from app.utils.env import resolve_env
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChatService:
    """聊天服务 - 整合所有组件，延迟初始化"""

    def __init__(self):
        settings = get_settings()

        # LLM
        from app.core import create_llm_service
        self.llm_service = create_llm_service(
            provider=settings.llm.provider,
            config={
                "api_key": settings.llm.api_key,
                "model": settings.llm.model,
                "max_tokens": settings.llm.max_tokens,
                "temperature": settings.llm.temperature,
                "base_url": settings.llm.base_url,
            },
        )

        # 记忆服务
        from app.memory import ConversationMemoryService, MySQLMemoryStrategy
        memory_strategy = MySQLMemoryStrategy(
            host=settings.mysql.host,
            port=settings.mysql.port,
            username=settings.mysql.username,
            password=settings.mysql.password,
            database=settings.mysql.database,
            recent_turns=settings.memory.recent_turns,
            recent_max_chars=settings.memory.recent_max_chars,
            summary_max_turns=settings.memory.summary_max_turns,
            summary_max_chars=settings.memory.summary_max_chars,
            llm_service=self.llm_service,
        )
        self.memory_service = ConversationMemoryService(memory_strategy)

        # 向量存储
        from app.core.chroma_client import create_vector_store
        vector_cfg = {
            "persist_directory": settings.chroma.persist_directory,
            "collection_name": settings.chroma.collection_name,
        }
        self.vector_store = create_vector_store(vector_cfg)

        # 关键词检索通道 (Elasticsearch)
        from elasticsearch import Elasticsearch
        from app.retrieval.pipeline import KeywordRetrievalChannel as ESKeywordRetrievalChannel
        es_hosts = [resolve_env(h) if isinstance(h, str) else h for h in settings.elasticsearch.hosts]
        self.es_client = Elasticsearch(hosts=es_hosts)
        self.keyword_channel = ESKeywordRetrievalChannel(
            elasticsearch_client=self.es_client,
            config={
                "index": settings.elasticsearch.index,
                "min_keyword_score": settings.retrieval.min_keyword_score,
            },
        )

        # Rerank 服务
        from app.core.rerank_service import SiliconFlowRerankService
        if settings.rerank.api_key:
            self.rerank_service = SiliconFlowRerankService(
                api_key=settings.rerank.api_key,
                model=settings.rerank.model,
            )
        else:
            self.rerank_service = None

        # RAG 检索引擎
        from app.retrieval import RagRetrievalEngine, VectorRetrievalChannel
        retrieval_dict = {
            "rrf_k": settings.retrieval.rrf_k,
            "max_parent_chunks": settings.retrieval.max_parent_chunks,
            "evidence_budget_per_child": settings.retrieval.evidence_budget_per_child,
            "evidence_budget_total": settings.retrieval.evidence_budget_total,
            "enable_semantic_compress": settings.retrieval.enable_semantic_compress,
            "vector_top_k": settings.retrieval.vector_top_k,
            "keyword_top_k": settings.retrieval.keyword_top_k,
            "min_vector_similarity": settings.retrieval.min_vector_similarity,
            "min_keyword_score": settings.retrieval.min_keyword_score,
        }
        self.rag_engine = RagRetrievalEngine(
            vector_channel=VectorRetrievalChannel(
                embedding_service=self.llm_service,
                vector_store=self.vector_store,
                config=retrieval_dict,
            ),
            keyword_channel=self.keyword_channel,
            rerank_service=self.rerank_service,
            llm_service=self.llm_service,
            config=retrieval_dict,
        )

        # 图查询引擎 (Neo4j)
        from app.core.graph_engine import create_graph_engine
        self.graph_query_engine = None
        if settings.neo4j.uri:
            try:
                self.graph_query_engine = create_graph_engine(
                    provider="neo4j",
                    config={
                        "neo4j": {
                            "uri": settings.neo4j.uri,
                            "username": settings.neo4j.username,
                            "password": settings.neo4j.password,
                        }
                    },
                )
                logger.info("图查询引擎初始化成功")
            except Exception as e:
                logger.warning(f"图查询引擎初始化失败: {e}")

        # Prompt 组装
        from app.core import RagPromptAssemblyService
        self.prompt_assembler = RagPromptAssemblyService(config={
            "per_sub_question_budget": settings.retrieval.evidence_budget_per_child,
            "total_budget": settings.retrieval.evidence_budget_total,
        })

        # 文档服务 & 知识路由
        from app.knowledge import KnowledgeRouteService, DocumentQuestionRouter, DocumentService
        self.document_service = DocumentService(vector_store=self.vector_store)
        self.knowledge_route_service = KnowledgeRouteService(self.llm_service, self.document_service)
        self.document_router = DocumentQuestionRouter(graph_query_engine=self.graph_query_engine)

        # 编排器
        from app.rag import ChatPreparationOrchestrator
        self.orchestrator = ChatPreparationOrchestrator(
            llm_service=self.llm_service,
            memory_service=self.memory_service,
            document_router=self.document_router,
            knowledge_route_service=self.knowledge_route_service,
            document_service=self.document_service,
            config=retrieval_dict,
        )

        # ReAct Agent
        from app.agent.react import ReActAgent, DuckDuckGoSearchTool, GetCurrentTimeTool
        from app.agent.skill import get_skill_manager
        tools = []

        try:
            ddg_tool = DuckDuckGoSearchTool(max_results=settings.search.max_results)
            tools.append(ddg_tool)
            skill_manager = get_skill_manager()
            skill_manager.register_tool("web_search", ddg_tool)
            logger.info(f"DuckDuckGo search tool registered, max_results={settings.search.max_results}")
        except ImportError as e:
            logger.warning(f"DuckDuckGo not available: {e}")

        time_tool = GetCurrentTimeTool()
        tools.append(time_tool)
        skill_manager = get_skill_manager()
        skill_manager.register_tool("get_current_time", time_tool)
        logger.info("GetCurrentTime tool registered")

        skill_cfgs = [s.model_dump() for s in settings.skills]
        skill_manager.load_skills_from_config(skill_cfgs)
        logger.info(f"Loaded {len(skill_manager.list_skills())} skills")

        self.react_agent = ReActAgent(self.llm_service, tools, {
            "model_call_limit": settings.agent.model_call_limit,
            "tool_call_limit": settings.agent.tool_call_limit,
            "session_model_call_limit": settings.agent.session_model_call_limit,
            "session_tool_call_limit": settings.agent.session_tool_call_limit,
        })

    async def chat(self, request):
        """处理聊天请求"""
        from app.models.chat import ChatResponse, SourceReference, ExecutionMode
        from app.agent import ClarificationExecutor, AgentExecutor

        logger.info(f"CHAT: question={request.question}, mode={request.chat_mode}")

        try:
            conversation_id = request.conversation_id or f"conv_{id(request)}"

            task_info = {
                "conversation_id": conversation_id,
                "question": request.question,
                "chat_mode": request.chat_mode.value,
                "selected_document_id": request.selected_document_id,
                "selected_document_name": request.selected_document_name,
                "selected_task_id": request.selected_task_id,
                "current_date": None,
                "current_date_text": "",
                "user_id": request.user_id,
            }

            plan = await self.orchestrator.prepare(task_info)
            logger.debug(f"Plan mode: {plan.mode}")

            if plan.mode == ExecutionMode.CLARIFICATION:
                executor = ClarificationExecutor(self.llm_service)
                result = await executor.execute({
                    "question": plan.original_question,
                    "clarification_reply": plan.clarification_reply,
                    "clarification_options": plan.clarification_options,
                })

            elif plan.mode == ExecutionMode.REACT_AGENT:
                executor = AgentExecutor(self.react_agent)
                memory_context = await self.memory_service.load_memory_context(
                    conversation_id, request.user_id
                )
                history_text = memory_context.get("recent_transcript", "") or ""
                history = []
                if history_text:
                    for line in history_text.split("\n"):
                        if line.startswith("用户:"):
                            history.append({"role": "user", "content": line[3:].strip()})
                        elif line.startswith("助手:"):
                            history.append({"role": "assistant", "content": line[3:].strip()})
                result = await executor.execute({
                    "question": plan.original_question,
                    "history": history,
                })

            else:
                result = await self._handle_rag_mode(plan, request.user_id)

            # 保存对话记忆
            content = result.content if hasattr(result, 'content') else str(result)
            await self._save_conversation(conversation_id, request.user_id, request.question, content)

            # 构建来源
            sources = self._build_sources(result)

            return ChatResponse(
                answer=str(content) if content else "无内容",
                sources=sources,
                suggested_questions=getattr(result, 'suggested_questions', []) or [],
                conversation_id=conversation_id,
            )

        except Exception as e:
            import traceback
            err_msg = f"处理出错: {e}\n{traceback.format_exc()}"
            logger.error(err_msg)
            return ChatResponse(
                answer=err_msg,
                sources=[],
                suggested_questions=[],
                conversation_id=request.conversation_id or "error",
            )

    async def _handle_rag_mode(self, plan, user_id: int):
        """处理 RAG 检索模式"""
        from app.models.chat import ExecutionMode

        retrieval_context = None
        answer = ""

        if plan.navigation_decision and plan.navigation_decision.execution_mode == ExecutionMode.GRAPH_THEN_EVIDENCE:
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
                    retrieval_context = await self.rag_engine.retrieve(plan, user_id=user_id)
                    if retrieval_context.is_empty():
                        answer = await self.llm_service.chat(
                            plan.original_question,
                            system_prompt="你是一个智能助手，请直接回答用户问题。",
                        )
                    else:
                        prompt_dict = self.prompt_assembler.assemble(plan, retrieval_context)
                        answer = await self.llm_service.chat(
                            prompt_dict["user_prompt"],
                            system_prompt=prompt_dict["system_prompt"],
                        )
            else:
                retrieval_context = await self.rag_engine.retrieve(plan, user_id=user_id)
                if retrieval_context.is_empty():
                    answer = await self.llm_service.chat(
                        plan.original_question,
                        system_prompt="你是一个智能助手，请直接回答用户问题。",
                    )
                else:
                    prompt_dict = self.prompt_assembler.assemble(plan, retrieval_context)
                    answer = await self.llm_service.chat(
                        prompt_dict["user_prompt"],
                        system_prompt=prompt_dict["system_prompt"],
                    )
        else:
            retrieval_context = await self.rag_engine.retrieve(plan, user_id=user_id)
            if retrieval_context.is_empty():
                logger.info("RAG 无证据，降级为 LLM 自由回答")
                answer = await self.llm_service.chat(
                    plan.original_question,
                    system_prompt="你是一个智能助手。知识库中没有找到相关信息，请基于你的知识直接回答用户问题。",
                )
            else:
                prompt_dict = self.prompt_assembler.assemble(plan, retrieval_context)
                logger.debug(f"LLM prompt length: user={len(prompt_dict['user_prompt'])}, system={len(prompt_dict['system_prompt'])}")
                answer = await self.llm_service.chat(
                    prompt_dict["user_prompt"],
                    system_prompt=prompt_dict["system_prompt"],
                )

        sources = self._build_sources(retrieval_context)

        class RAGResult:
            def __init__(self):
                self.mode = ExecutionMode.RETRIEVAL
                self.success = True
                self.content = answer
                self.sources = sources
                self.suggested_questions = []

        return RAGResult()

    def _build_sources(self, result_or_context) -> list:
        """构建来源列表"""
        sources = []
        try:
            refs = None
            if hasattr(result_or_context, 'sources') and result_or_context.sources:
                refs = result_or_context.sources
            elif hasattr(result_or_context, 'flatten_references'):
                refs = result_or_context.flatten_references()

            if refs:
                from app.models.chat import SourceReference
                for ref in refs[:5]:
                    if isinstance(ref, dict):
                        sources.append({
                            "reference_id": str(ref.get("reference_id", "")),
                            "document_name": str(ref.get("document_name", "")),
                            "section_path": str(ref.get("section_path", "")),
                            "channel": str(ref.get("channel", "")),
                        })
        except Exception as e:
            logger.warning(f"构建来源失败: {e}")
        return sources

    async def _save_conversation(self, conversation_id: str, user_id: int, question: str, answer: str):
        """保存对话记忆和会话记录"""
        try:
            await self.memory_service.save_exchange(conversation_id, user_id, question, answer)

            settings = get_settings()
            conn = pymysql.connect(
                host=settings.mysql.host,
                port=settings.mysql.port,
                user=settings.mysql.username,
                password=settings.mysql.password,
                database=settings.mysql.database,
                charset="utf8mb4",
            )
            cur = conn.cursor()
            cur.execute("""
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS message (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id VARCHAR(64) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_conversation_id (conversation_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            conn.commit()
            cur.execute(
                "INSERT INTO conversation (id, user_id, title, summary) VALUES (%s, %s, %s, '') "
                "ON DUPLICATE KEY UPDATE updated_at = NOW(), user_id = COALESCE(user_id, VALUES(user_id))",
                (conversation_id, user_id or 0, question[:100]),
            )
            cur.execute("INSERT INTO message (conversation_id, role, content) VALUES (%s, 'user', %s)",
                        (conversation_id, question))
            cur.execute("INSERT INTO message (conversation_id, role, content) VALUES (%s, 'assistant', %s)",
                        (conversation_id, answer))
            conn.commit()
            logger.info(f"CONV SAVE: id={conversation_id[:12]}, user_id={user_id}")
            conn.close()
        except Exception as e:
            logger.warning(f"保存记忆失败: {e}")


# 全局单例（延迟初始化）
_chat_service = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
