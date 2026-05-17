"""
RAG前置编排引擎
包含：路由判定、问题改写、子问题拆分、意图解析、歧义检测
对应Java的ChatPreparationOrchestrator
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import re

from app.core.logging import get_logger
from app.models.chat import ExecutionMode  # 统一从 models 导入

logger = get_logger(__name__)


@dataclass
class SubQuestion:
    """子问题"""
    index: int
    question: str
    knowledge_domain: str = ""
    rewritten_question: str = ""
    confidence: float = 1.0


@dataclass
class RetrievalQuestionPlan:
    """检索问题计划"""
    retrieval_question: str
    sub_questions: List[str] = field(default_factory=list)


@dataclass
class DocumentNavigationDecision:
    """文档导航决策"""
    execution_mode: ExecutionMode
    retrieval_plan: Optional[RetrievalQuestionPlan] = None
    structure_anchor: Optional[Dict[str, Any]] = None
    item_anchor: Optional[Dict[str, Any]] = None
    summary_text: str = ""


@dataclass
class ConversationExecutionPlan:
    """会话执行计划"""
    chat_mode: str
    original_question: str
    agent_question: str = ""
    rewrite_question: str = ""
    rewrite_sub_questions: List[str] = field(default_factory=list)
    retrieval_question: str = ""
    retrieval_sub_questions: List[str] = field(default_factory=list)
    history_summary: str = ""
    long_term_summary: str = ""
    recent_history_transcript: str = ""
    answer_recent_transcript: str = ""
    mode: ExecutionMode = ExecutionMode.RETRIEVAL
    navigation_decision: Optional[DocumentNavigationDecision] = None
    selected_document_id: Optional[int] = None
    selected_document_name: Optional[str] = None
    selected_task_id: Optional[int] = None
    retrieval_document_ids: List[str] = field(default_factory=list)  # UUID 字符串列表
    retrieval_task_ids: List[int] = field(default_factory=list)
    no_evidence_reply: str = "当前没有足够证据支持明确回答。"
    clarification_reply: str = ""
    clarification_options: List[str] = field(default_factory=list)
    clarification_reason: str = ""
    requires_current_date_anchoring: bool = False
    requires_fresh_search: bool = False


@dataclass
class RagRewriteResult:
    """RAG改写结果"""
    rewritten_question: str
    sub_questions: List[str] = field(default_factory=list)
    raw_model_output: str = ""


class TimeSensitiveQueryHelper:
    """时间敏感查询帮助器"""

    @staticmethod
    def requires_current_date_anchoring(question: str) -> bool:
        """检查是否需要当前日期锚定"""
        current_date_keywords = ["今天", "明天", "昨天", "本周", "本月", "今年"]
        return any(kw in question for kw in current_date_keywords)

    @staticmethod
    def requires_fresh_search(question: str) -> bool:
        """检查是否需要实时搜索"""
        fresh_search_keywords = ["天气", "新闻", "股价", "汇率", "热搜", "最新", "现在"]
        return any(kw in question for kw in fresh_search_keywords)


class ChatPreparationOrchestrator:
    """RAG前置编排器 - 对标Java的ChatPreparationOrchestrator"""

    CAPABILITY_HINTS = {"你都能干什么", "你能做什么", "你可以做什么", "你会什么", "你是谁", "怎么用你", "能帮我什么"}
    OPEN_CHAT_HINTS = {"天气", "温度", "下雨", "新闻", "股价", "汇率", "热搜", "今天", "明天", "最新", "现在"}
    CHITCHAT_HINTS = {"你好", "您好", "hello", "hi", "谢谢", "感谢", "再见", "拜拜"}

    def __init__(self, llm_service, memory_service, document_router, knowledge_route_service, document_service, config: Dict[str, Any]):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.document_router = document_router
        self.knowledge_route_service = knowledge_route_service
        self.document_service = document_service
        self.config = config
        self.no_evidence_reply = config.get("no_evidence_reply", "当前没有足够证据支持明确回答。")

    async def prepare(self, task_info: Dict[str, Any]) -> ConversationExecutionPlan:
        """执行完整的编排流程 - 对标Java的prepare方法"""
        conversation_id = task_info["conversation_id"]
        question = task_info["question"]
        chat_mode = task_info["chat_mode"]
        selected_document_id = task_info.get("selected_document_id")
        selected_document_name = task_info.get("selected_document_name")
        selected_task_id = task_info.get("selected_task_id")
        current_date = task_info.get("current_date")
        current_date_text = task_info.get("current_date_text", "")

        # 1. 装载会话记忆
        memory_context = await self.memory_service.load_memory_context(conversation_id, task_info.get("user_id"))
        history_summary = self._build_history_summary(memory_context)

        # 1.5 检测用户是否在回复澄清列表（选择文档）
        _routed_doc_uuid = None  # 缓存 UUID 避免重复查询
        clarification_selection = self._parse_clarification_selection(question)
        if clarification_selection and chat_mode == "AUTO_DOCUMENT":
            doc_info = await self._resolve_document_id_by_name(clarification_selection)
            if doc_info:
                logger.debug(f"[ORCH] User selected document from clarification: {clarification_selection}")
                selected_document_name = doc_info["document_name"]
                selected_document_id = doc_info["int_id"]
                _routed_doc_uuid = doc_info["uuid"]
                chat_mode = "DOCUMENT"
            else:
                logger.debug(f"[ORCH] Document name '{clarification_selection}' not found in knowledge base")

        # 1.6 检测追问并通过记忆上下文继承文档选择
        if self._is_followup_question(question, history_summary) and chat_mode == "AUTO_DOCUMENT":
            followup_ctx = await self._resolve_followup_context(conversation_id, task_info.get("user_id"))
            if followup_ctx and followup_ctx.get("last_document_name"):
                doc_info = await self._resolve_document_id_by_name(followup_ctx["last_document_name"])
                if doc_info:
                    logger.debug(f"[ORCH] Follow-up question detected, inheriting document: {doc_info['document_name']}")
                    selected_document_name = doc_info["document_name"]
                    selected_document_id = doc_info["int_id"]
                    _routed_doc_uuid = doc_info["uuid"]
                    chat_mode = "DOCUMENT"

        # 2. AI自动判断意图 - LLM决定执行模式
        requires_current_date_anchoring = TimeSensitiveQueryHelper.requires_current_date_anchoring(question)
        requires_fresh_search = TimeSensitiveQueryHelper.requires_fresh_search(question)

        logger.debug(f"[INTENT DEBUG] question={question}, requires_fresh_search={requires_fresh_search}")
        intent_decision = await self._auto_detect_intent(question, history_summary, requires_fresh_search)
        logger.debug(f"[INTENT DEBUG] intent_decision={intent_decision}")

        # 如果LLM判断为自由聊天/工具调用，直接走ReAct Agent
        if intent_decision == "REACT_AGENT":
            logger.debug("[INTENT DEBUG] -> REACT_AGENT path")
            return self._base_plan(question, chat_mode, memory_context, history_summary, current_date, current_date_text,
                                    requires_current_date_anchoring, requires_fresh_search,
                                    mode=ExecutionMode.REACT_AGENT)

        # 如果LLM判断为闲聊，生成友好的闲聊回复（不走Agent，避免"思考: 无 动作: 无"）
        if intent_decision == "CHITCHAT":
            logger.debug("[INTENT DEBUG] -> CHITCHAT path")
            chitchat_reply = self._generate_chitchat_reply(question, history_summary)
            return self._base_plan(question, chat_mode, memory_context, history_summary, current_date, current_date_text,
                                    requires_current_date_anchoring, requires_fresh_search,
                                    mode=ExecutionMode.CLARIFICATION,
                                    chitchat_reply=chitchat_reply)

        # 意图判断为RAG后，继续执行知识路由
        # 如果意图判断失败，默认也走知识路由

        # 3. 问题改写
        rewrite_result = await self._rewrite_question(question, history_summary)
        rewrite_question = rewrite_result.rewritten_question if rewrite_result else question
        rewrite_sub_questions = rewrite_result.sub_questions if rewrite_result and rewrite_result.sub_questions else [rewrite_question]

        # 4. 知识路由（AUTO_DOCUMENT模式）
        routed_document_id = selected_document_id
        routed_document_name = selected_document_name
        routed_task_id = selected_task_id
        routed_document_ids = [routed_document_id] if routed_document_id else []
        routed_task_ids = [routed_task_id] if routed_task_id else []

        # 如果有文档名但没有文档ID，从文档服务获取ID
        if chat_mode == "DOCUMENT" and selected_document_name and not selected_document_id:
            doc_info = await self._resolve_document_id_by_name(selected_document_name)
            if doc_info:
                routed_document_id = doc_info["int_id"]
                routed_document_name = doc_info["document_name"]
                routed_document_ids = [doc_info["uuid"]]
                routed_task_ids = []
                _routed_doc_uuid = doc_info["uuid"]

        # 如果有文档ID但没有UUID，解析UUID（优先使用缓存的UUID）
        if chat_mode == "DOCUMENT" and selected_document_id and not routed_document_ids:
            uuid = _routed_doc_uuid or await self._resolve_document_uuid(selected_document_id)
            if uuid:
                routed_document_ids = [uuid]

        # 如果用户指定了文档名在前端（DOCUMENT模式下 selected_document_name 有值）
        if chat_mode == "DOCUMENT" and selected_document_name and selected_document_id:
            uuid = _routed_doc_uuid or await self._resolve_document_uuid(selected_document_id)
            if uuid:
                routed_document_ids = [uuid]

        if chat_mode == "AUTO_DOCUMENT":
            route_decision = await self.knowledge_route_service.route(question, rewrite_question)

            # 检查是否需要澄清
            if self._should_ask_clarification(route_decision):
                clarification_reply = self._build_clarification_reply(question, route_decision)
                return self._base_plan(question, chat_mode, memory_context, history_summary, current_date, current_date_text,
                                       requires_current_date_anchoring, requires_fresh_search,
                                       mode=ExecutionMode.CLARIFICATION,
                                       rewrite_question=rewrite_question,
                                       rewrite_sub_questions=rewrite_sub_questions,
                                       retrieval_question=rewrite_question,
                                       retrieval_sub_questions=rewrite_sub_questions,
                                       clarification_reply=clarification_reply)

            # 获取最高置信度文档
            if route_decision and route_decision.documents:
                top_doc = route_decision.documents[0]
                doc_id = top_doc.document_id
                # routed_document_id 保持 int 用于图查询
                if isinstance(doc_id, str) and len(doc_id) == 36 and '-' in doc_id:
                    routed_document_id = int(doc_id[:8], 16)
                else:
                    routed_document_id = int(doc_id) if doc_id else None
                routed_document_name = top_doc.document_name
                task_id = top_doc.last_index_task_id
                routed_task_id = int(task_id) if task_id and str(task_id).isdigit() else None

            # 处理文档ID列表 — 保持 UUID 字符串用于检索过滤
            routed_document_ids = []
            for doc in route_decision.documents:
                doc_id = doc.document_id
                # 保持原始 UUID 字符串，不做 int 转换
                routed_document_ids.append(str(doc_id) if doc_id else "")
            routed_task_ids = [int(doc.last_index_task_id) if doc.last_index_task_id and str(doc.last_index_task_id).isdigit() else None for doc in route_decision.documents]

        # 5. 文档导航决策
        navigation_decision = await self.document_router.route(routed_document_id, question, rewrite_result)

        execution_mode = navigation_decision.execution_mode if navigation_decision else ExecutionMode.RETRIEVAL

        # 安全访问 retrieval_plan
        if navigation_decision and navigation_decision.retrieval_plan:
            retrieval_question = navigation_decision.retrieval_plan.retrieval_question
            retrieval_sub_questions = navigation_decision.retrieval_plan.sub_questions or rewrite_sub_questions
        else:
            retrieval_question = rewrite_question
            retrieval_sub_questions = rewrite_sub_questions

        # 6. 构建执行计划
        return self._base_plan(question, chat_mode, memory_context, history_summary, current_date, current_date_text,
                               requires_current_date_anchoring, requires_fresh_search,
                               mode=execution_mode,
                               rewrite_question=rewrite_question,
                               rewrite_sub_questions=rewrite_sub_questions,
                               retrieval_question=retrieval_question,
                               retrieval_sub_questions=retrieval_sub_questions,
                               navigation_decision=navigation_decision,
                               selected_document_id=routed_document_id,
                               selected_document_name=routed_document_name,
                               selected_task_id=routed_task_id,
                               retrieval_document_ids=routed_document_ids,
                               retrieval_task_ids=routed_task_ids,
                               no_evidence_reply=self._build_no_evidence_reply(question, requires_fresh_search, navigation_decision))

    async def _resolve_document_uuid(self, doc_id_int: int) -> Optional[str]:
        """将前端传入的 int ID 解析为 Chroma 中的 UUID 字符串"""
        documents = await self.document_service.list_retrievable_documents()
        for doc in documents:
            doc_uuid = doc.get("id", "")
            if isinstance(doc_uuid, str) and len(doc_uuid) == 36:
                try:
                    if int(doc_uuid[:8], 16) == doc_id_int:
                        return doc_uuid
                except ValueError:
                    pass
        return None

    def _parse_clarification_selection(self, question: str) -> Optional[str]:
        """解析用户从澄清列表中选择的文档名"""
        # 匹配 "1. 《文档名》" 或 "《文档名》" 格式
        import re
        # 选项编号 + 书名号
        m = re.search(r'[\d]+[\.\、\s]*《([^》]+)》', question)
        if m:
            return m.group(1).strip()
        # 纯书名号
        m = re.search(r'《([^》]+)》', question)
        if m:
            return m.group(1).strip()
        return None

    def _is_followup_question(self, question: str, history_summary: str) -> bool:
        """检测是否为上下文依赖的追问"""
        followup_patterns = [
            r'^为什么\s*$', r'^为啥\s*$', r'^怎么说\s*$', r'^什么意思\s*$',
            r'^详细[说说讲]\s*$', r'^具体[点些]\s*$', r'^然后呢\s*$',
            r'^继续\s*$', r'^还有呢\s*$', r'^然后\s*$',
            r'^那[这个些]?呢\s*$', r'^怎么[做搞]\s*$',
            r'^举个例子\s*$', r'^比如\s*$',
        ]
        clean = question.strip()
        for pattern in followup_patterns:
            if re.match(pattern, clean):
                return True
        return False

    async def _resolve_followup_context(self, conversation_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """从记忆服务中获取上一轮的文档上下文"""
        memory_context = await self.memory_service.load_memory_context(conversation_id, user_id)
        recent = memory_context.get("recent_transcript", "") if memory_context else ""
        if not recent:
            return None
        # 从最近对话中提取文档名
        import re
        doc_names = re.findall(r'《([^》]+)》', recent)
        if doc_names:
            # 取最后出现的文档名
            return {"last_document_name": doc_names[-1], "has_context": True}
        return None

    async def _resolve_document_id_by_name(self, document_name: str) -> Optional[Dict[str, Any]]:
        """通过文档名查找文档 UUID 和 int ID"""
        documents = await self.document_service.list_retrievable_documents()
        for doc in documents:
            if doc.get("document_name") == document_name:
                doc_uuid = doc.get("id", "")
                doc_int_id = None
                if isinstance(doc_uuid, str) and len(doc_uuid) == 36:
                    try:
                        doc_int_id = int(doc_uuid[:8], 16)
                    except ValueError:
                        pass
                return {
                    "uuid": doc_uuid,
                    "int_id": doc_int_id,
                    "document_name": document_name
                }
        return None

    async def _rewrite_question(self, question: str, history_summary: str) -> RagRewriteResult:
        """问题改写 - 对标Java的ChatQueryRewriteService"""
        prompt = f"""将用户的问题改写为检索友好的表达。

原始问题: {question}
历史上下文: {history_summary}

要求:
1. 补全省略的主语、指代等
2. 补充会话中的上下文信息
3. 如果问题涉及多个子问题，拆分为独立子问题列表
4. 保持原意不变

请以JSON格式返回:
{{"rewritten_question": "...", "sub_questions": ["...", "..."]}}
"""
        response = await self.llm_service.chat(prompt)
        # 简单解析，实际应该用结构化输出
        try:
            import json
            result = json.loads(response)
            return RagRewriteResult(
                rewritten_question=result.get("rewritten_question", question),
                sub_questions=result.get("sub_questions", []),
                raw_model_output=response
            )
        except:
            return RagRewriteResult(rewritten_question=question, sub_questions=[question], raw_model_output=response)

    def _should_ask_clarification(self, route_decision) -> bool:
        """判断是否需要澄清"""
        if not route_decision or not route_decision.documents:
            return True
        # 去重：按文档名去重
        seen_names = set()
        unique_docs = []
        for doc in route_decision.documents:
            if doc.document_name not in seen_names:
                seen_names.add(doc.document_name)
                unique_docs.append(doc)
        route_decision.documents = unique_docs

        # 置信度太低，需要澄清
        if route_decision.confidence < 0.4:
            return True
        # 只有一个文档，直接使用
        if len(route_decision.documents) < 2:
            return False
        top_score = route_decision.documents[0].score
        # 最高分太低，需要澄清
        if top_score < 0.35:
            return True
        second_score = route_decision.documents[1].score
        # 前两名差距太小，说明存在歧义
        if top_score > 0 and (top_score - second_score) <= 0.05:
            return True
        return False

    def _build_clarification_reply(self, question: str, route_decision) -> str:
        """构建澄清回复"""
        candidates = route_decision.documents[:5] if route_decision.documents else []
        if not candidates:
            return "当前我还不能稳定判断你想问哪份知识文档。请补充更具体的文档名或主题词。"

        # 去重：按文档名去重
        seen_names = set()
        unique_candidates = []
        for doc in candidates:
            if doc.document_name not in seen_names:
                seen_names.add(doc.document_name)
                unique_candidates.append(doc)
        candidates = unique_candidates[:3]

        if len(candidates) == 1:
            return ""

        builder = "这个问题可能涉及多份文档，你想问哪一份？\n"
        for i, doc in enumerate(candidates):
            builder += f"{i+1}. 《{doc.document_name}》"
            if doc.knowledge_scope_name:
                builder += f"（{doc.knowledge_scope_name}）"
            builder += "\n"
        builder += "你可以直接回复文档名或编号。"
        return builder

    def _build_no_evidence_reply(self, question: str, requires_fresh_search: bool, navigation_decision=None) -> str:
        """构建无证据回复"""
        # 如果有图查询结果，返回图查询内容
        if navigation_decision and navigation_decision.structure_anchor:
            graph_data = navigation_decision.structure_anchor.get("graph_data", {})
            if graph_data.get("status") == "success":
                if graph_data.get("type") == "structure":
                    chapters = graph_data.get("chapters", [])
                    if chapters:
                        content = f"《{graph_data.get('document_title', '')}》包含以下章节：\n"
                        for ch in chapters:
                            content += f"- {ch}\n"
                        return content
                elif graph_data.get("type") == "section":
                    return f"{graph_data.get('chapter_title', '')}：{graph_data.get('chapter_content', '')}"
                elif graph_data.get("type") == "item":
                    return f"{graph_data.get('item_title', '')}：{graph_data.get('item_content', '')}"

        if self._looks_like_capability_question(question):
            return "当前你正在使用\"当前文档问答\"模式，这个问题更像是在询问助手能力。如果你想了解我能做什么，请切换到\"开放式提问\"模式。"
        if self._looks_like_open_chat_question(question, requires_fresh_search):
            return "当前你正在使用\"当前文档问答\"模式，这个问题更像开放式提问。如果你想继续问这类问题，请切换到\"开放式提问\"模式。"
        return self.no_evidence_reply

    async def _auto_detect_intent(self, question: str, history_summary: str, requires_fresh_search: bool) -> str:
        """AI自动判断意图 - 决定走RAG还是ReAct Agent"""
        logger.debug(f"[INTENT] Analyzing: {question}")

        capability_hints = ["你都能干什么", "你能做什么", "你可以做什么", "你会什么", "你是谁", "怎么用你", "能帮我什么"]
        chitchat_hints = ["你好", "您好", "hello", "hi", "谢谢", "感谢", "再见", "拜拜", "在吗", "干嘛", "你好", "嗨"]
        tool_needed_hints = ["搜索", "查询", "最新", "现在", "今日", "天气", "新闻", "股价", "汇率", "实时", "今天几", "现在几", "几点", "几号", "星期几", "时间", "日期"]

        # 清理问题中的标点符号
        clean = question.strip().rstrip('?').rstrip('？').rstrip('!').rstrip('！').rstrip('.').rstrip('。').strip()
        clean_lower = clean.lower()

        # 0. 明确的中文寒暄（精确匹配，去除标点后）
        if clean_lower in ["你好", "您好", "hi", "hello", "嗨", "嘿"]:
            logger.debug("[INTENT] -> CHITCHAT (greeting)")
            return "CHITCHAT"

        # 1. 检查闲聊关键词
        for hint in chitchat_hints:
            if hint in clean:
                logger.debug(f"[INTENT] -> CHITCHAT (hint: {hint})")
                return "CHITCHAT"

        # 2. 检查闲聊（简单符号/单字）
        if clean in ["1", "。", "", "啊", "嗯", "哦", "呃"]:
            logger.debug("[INTENT] -> CHITCHAT (simple)")
            return "CHITCHAT"

        # 3. 检查是否需要工具/实时信息
        if requires_fresh_search or any(hint in clean for hint in tool_needed_hints):
            logger.debug(f"[INTENT] -> REACT_AGENT (needs tools or fresh search)")
            return "REACT_AGENT"

        # 4. 检查是否询问能力
        if any(hint in clean for hint in capability_hints):
            logger.debug("[INTENT] -> CHITCHAT (capability)")
            return "CHITCHAT"

        # 5. 文档名直接匹配（书名号或扩展名）
        if ".md" in clean or ".doc" in clean or ".pdf" in clean or "《" in clean:
            logger.debug("[INTENT] -> RAG (document reference)")
            return "RAG"

        # 5.1 编号选择（来自澄清列表的回复，如 "1"、"第1个"、"选1"）
        if re.match(r'^[\d]+[\.\、\s]?$', clean) or re.match(r'^[第]?[\d]+[个份]$', clean) or re.match(r'^[选]?[\d]+$', clean):
            logger.debug("[INTENT] -> RAG (numbered selection)")
            return "RAG"

        # 6. 通用短问题（5字以内，无字母）走闲聊
        if len(clean) <= 5 and not any(c.isalpha() for c in clean):
            logger.debug(f"[INTENT] -> CHITCHAT (short generic)")
            return "CHITCHAT"

        logger.debug("[INTENT] -> RAG (default)")
        return "RAG"

    def _needs_llm_intent_classification(self, question: str, history_summary: str) -> bool:
        """判断是否需要LLM辅助判断意图"""
        # 通用问题可能是闲聊也可能是知识问答
        short_generic = len(question) < 10 and not any(c.isalpha() for c in question)
        if short_generic:
            return True
        # 有历史记录时，如果用户没有明确指明文档，可能需要判断
        if history_summary and not any(kw in question for kw in ["文档", "文章", "第", "章", "哪个"]):
            return True
        return False

    async def _llm_classify_intent(self, question: str, history_summary: str) -> str:
        """LLM辅助判断意图"""
        prompt = f"""判断用户问题的意图类型。

用户问题: {question}
对话历史: {history_summary or '无'}

请判断这个问题的意图：
- 如果是闲聊、寒暄、无需检索的问题，返回 "CHITCHAT"
- 如果是需要搜索网络、实时信息的问题，返回 "REACT_AGENT"
- 如果是需要检索知识库文档的问题，返回 "RAG"

只返回一种类型，不要解释。"""

        try:
            response = await self.llm_service.chat(prompt)
            response = response.strip().upper()
            if "CHITCHAT" in response:
                return "CHITCHAT"
            elif "REACT" in response:
                return "REACT_AGENT"
            else:
                return "RAG"
        except:
            return "RAG"

    def _looks_like_capability_question(self, question: str) -> bool:
        return any(hint in question for hint in self.CAPABILITY_HINTS)

    def _looks_like_open_chat_question(self, question: str, requires_fresh_search: bool) -> bool:
        if requires_fresh_search:
            return True
        return any(hint in question for hint in self.OPEN_CHAT_HINTS)

    def _generate_chitchat_reply(self, question: str, history_summary: str) -> str:
        """生成闲聊回复（用于寒暄、问候等场景）"""
        clean = question.strip().rstrip('?').rstrip('？').rstrip('!').rstrip('！').rstrip('.').rstrip('。').strip().lower()

        # 问候语
        if clean in ["你好", "您好", "hi", "hello", "嗨", "嘿"]:
            return "你好！我是智能助手，可以回答问题、查询知识、搜索信息等。有什么可以帮你的吗？"

        # 感谢
        if clean in ["谢谢", "感谢", "thanks", "thank you"]:
            return "不客气！很高兴能帮到你。有什么其他问题吗？"

        # 再见
        if clean in ["再见", "拜拜", "bye", "goodbye"]:
            return "再见！有需要随时找我。"

        # 询问能力
        for hint in self.CAPABILITY_HINTS:
            if hint in question:
                return "我是智能助手，主要功能包括：\n1. 知识问答：基于已上传的文档回答问题\n2. 联网搜索：查询最新资讯（天气、新闻、股价等）\n3. 文档对话：可以针对具体文档进行深入讨论\n\n试试问我一个问题吧！"

        # 短回复/无意义输入
        if len(clean) <= 5:
            return "你好！有什么我可以帮你的吗？"

        # 默认
        return "你好！我可以帮你回答问题、查询知识或搜索信息。有什么需要尽管问我！"

    def _build_history_summary(self, memory_context) -> str:
        """构建历史摘要"""
        if not memory_context:
            return ""
        recent = memory_context.get("recent_transcript", "")
        summary = memory_context.get("long_term_summary", "")
        return f"{summary}\n\n{recent}" if summary else recent

    def _base_plan(self, question: str, chat_mode: str, memory_context, history_summary: str,
                   current_date, current_date_text: str,
                   requires_current_date_anchoring: bool, requires_fresh_search: bool,
                   mode: ExecutionMode = ExecutionMode.RETRIEVAL,
                   rewrite_question: str = None,
                   rewrite_sub_questions: List[str] = None,
                   retrieval_question: str = None,
                   retrieval_sub_questions: List[str] = None,
                   navigation_decision=None,
                   selected_document_id=None,
                   selected_document_name=None,
                   selected_task_id=None,
                   retrieval_document_ids: List[int] = None,
                   retrieval_task_ids: List[int] = None,
                   no_evidence_reply: str = None,
                   clarification_reply: str = "",
                   clarification_options: List[str] = None,
                   chitchat_reply: str = None) -> ConversationExecutionPlan:
        """构建基础执行计划"""
        return ConversationExecutionPlan(
            chat_mode=chat_mode,
            original_question=question,
            agent_question=question,
            rewrite_question=rewrite_question or question,
            rewrite_sub_questions=rewrite_sub_questions or [question],
            retrieval_question=retrieval_question or question,
            retrieval_sub_questions=retrieval_sub_questions or [question],
            history_summary=history_summary,
            long_term_summary=memory_context.get("long_term_summary", "") if memory_context else "",
            recent_history_transcript=memory_context.get("recent_transcript", "") if memory_context else "",
            mode=mode,
            navigation_decision=navigation_decision,
            selected_document_id=selected_document_id,
            selected_document_name=selected_document_name,
            selected_task_id=selected_task_id,
            retrieval_document_ids=retrieval_document_ids or [],
            retrieval_task_ids=retrieval_task_ids or [],
            no_evidence_reply=no_evidence_reply or self.no_evidence_reply,
            clarification_reply=clarification_reply or chitchat_reply or "",
            clarification_options=clarification_options or [],
            requires_current_date_anchoring=requires_current_date_anchoring,
            requires_fresh_search=requires_fresh_search
        )