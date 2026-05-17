"""
三层执行器
包含：ClarificationExecutor、RagChatExecutor、ReActAgentExecutor
"""
from typing import List, Dict, Any, Optional, AsyncIterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import asyncio

from app.models.chat import ExecutionMode  # 统一从 models 导入


@dataclass
class ExecutorResult:
    """执行器结果"""
    mode: ExecutionMode
    success: bool
    content: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseExecutor(ABC):
    """执行器基类"""

    def __init__(self, name: str, mode: ExecutionMode):
        self.name = name
        self.mode = mode

    @abstractmethod
    async def execute(self, task_info: Dict[str, Any]) -> ExecutorResult:
        """执行"""
        pass

    async def execute_stream(self, task_info: Dict[str, Any]) -> AsyncIterator[str]:
        """流式执行"""
        result = await self.execute(task_info)
        yield result.content


class ClarificationExecutor(BaseExecutor):
    """歧义追问执行器 - 对标Java的ClarificationExecutor"""

    def __init__(self, llm_service):
        super().__init__("ClarificationExecutor", ExecutionMode.CLARIFICATION)
        self.llm_service = llm_service

    async def execute(self, task_info: Dict[str, Any]) -> ExecutorResult:
        """生成澄清问题"""
        question = task_info.get("question", "")
        clarification_reply = task_info.get("clarification_reply", "")
        clarification_options = task_info.get("clarification_options", [])

        if not clarification_reply:
            prompt = f"""用户的问题信息量不足，无法确定意图。请生成澄清问题。

用户问题: {question}

请生成1-2个问题来引导用户补充必要信息。"""

            clarification_reply = await self.llm_service.chat(prompt)

        return ExecutorResult(
            mode=ExecutionMode.CLARIFICATION,
            success=True,
            content=clarification_reply,
            metadata={"options": clarification_options}
        )


class RagChatExecutor(BaseExecutor):
    """RAG知识问答执行器 - 对标Java的RagChatExecutor"""

    def __init__(self, llm_service, retrieval_engine, prompt_assembly_service):
        super().__init__("RagChatExecutor", ExecutionMode.RETRIEVAL)
        self.llm_service = llm_service
        self.retrieval_engine = retrieval_engine
        self.prompt_assembly_service = prompt_assembly_service

    async def execute(self, task_info: Dict[str, Any]) -> ExecutorResult:
        """执行RAG知识问答"""
        plan = task_info.get("execution_plan")
        if not plan:
            return ExecutorResult(
                mode=ExecutionMode.RETRIEVAL,
                success=False,
                error="No execution plan"
            )

        # 执行检索
        retrieval_context = await self.retrieval_engine.retrieve(plan)

        # 检查是否有有效证据
        if retrieval_context.is_empty():
            no_evidence_reply = plan.no_evidence_reply or "当前没有足够证据支持明确回答。"
            return ExecutorResult(
                mode=ExecutionMode.RETRIEVAL,
                success=True,
                content=no_evidence_reply,
                sources=[]
            )

        # 组装Prompt
        prompt_result = self.prompt_assembly_service.assemble(plan, retrieval_context)

        # 调用模型生成
        answer = await self.llm_service.chat(prompt_result["user_prompt"], system_prompt=prompt_result["system_prompt"])

        # 生成推荐追问
        suggested = await self._generate_suggested_questions(plan.original_question, retrieval_context)

        return ExecutorResult(
            mode=ExecutionMode.RETRIEVAL,
            success=True,
            content=answer,
            sources=retrieval_context.flatten_references(),
            suggested_questions=suggested,
            metadata={
                "used_channels": retrieval_context.used_channels,
                "retrieval_notes": retrieval_context.retrieval_notes
            }
        )

    async def _generate_suggested_questions(self, question: str, retrieval_context) -> List[str]:
        """生成推荐追问问题"""
        prompt = f"""基于用户问题和检索结果，生成3个可继续追问的问题。

用户问题: {question}
检索到的内容: {[r.get('content', '')[:100] for r in retrieval_context.flatten_references()[:3]]}

要求：生成简洁、有深度的问题，帮助用户深入探索主题。
格式：1. 问题1  2. 问题2  3. 问题3"""

        response = await self.llm_service.chat(prompt)

        # 简单解析
        questions = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                q = line.lstrip("0123456789.、- ").strip()
                if q:
                    questions.append(q)
            if len(questions) >= 3:
                break

        return questions[:3]


class AgentExecutor(BaseExecutor):
    """Agent执行器 - 对标Java的ReactAgentExecutor"""

    def __init__(self, react_agent):
        super().__init__("AgentExecutor", ExecutionMode.REACT_AGENT)
        self.react_agent = react_agent

    async def execute(self, task_info: Dict[str, Any]) -> ExecutorResult:
        """执行Agent"""
        question = task_info.get("question", "")
        history = task_info.get("history", [])

        result = await self.react_agent.execute(question, history)

        return ExecutorResult(
            mode=ExecutionMode.REACT_AGENT,
            success=result.get("error") is None,
            content=result.get("answer", ""),
            metadata={
                "model_calls": result.get("model_calls", 0),
                "tool_calls": result.get("tool_calls", 0),
                "used_tools": result.get("used_tools", []),
                "trace": result.get("trace", {})
            },
            error=result.get("error")
        )


class ExecutorRegistry:
    """执行器注册中心 - 对标Java的ConversationExecutorRegistry"""

    def __init__(self):
        self._executors: Dict[ExecutionMode, BaseExecutor] = {}

    def register(self, executor: BaseExecutor):
        """注册执行器"""
        self._executors[executor.mode] = executor

    def get(self, mode: ExecutionMode) -> Optional[BaseExecutor]:
        """获取执行器"""
        return self._executors.get(mode)

    async def execute(self, mode: ExecutionMode, task_info: Dict[str, Any]) -> ExecutorResult:
        """执行指定模式的执行器"""
        executor = self.get(mode)
        if not executor:
            return ExecutorResult(
                mode=mode,
                success=False,
                error=f"No executor for mode {mode}"
            )
        return await executor.execute(task_info)