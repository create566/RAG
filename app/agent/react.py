"""
LangGraph Agent核心模块
基于 LangGraph StateGraph 实现 ReAct Agent，对标 Java 的 ReactAgentExecutor
"""
from typing import List, Dict, Any, Optional, Callable, TypedDict, Annotated, AsyncIterator
from enum import Enum
import operator
import asyncio
import json

from langgraph.graph import StateGraph, END, START


# ── State ──────────────────────────────────────────────

class ReActState(TypedDict):
    """LangGraph ReAct Agent 状态"""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    question: str
    model_calls: int
    tool_calls: int
    used_tools: Annotated[List[str], operator.add]
    trace_steps: Annotated[List[Dict], operator.add]
    pending_tool_name: str
    pending_tool_params: Dict
    final_answer: str
    error: str


# ── Tools ──────────────────────────────────────────────

class AgentTool:
    """Agent工具"""

    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    async def execute(self, **kwargs) -> str:
        """执行工具"""
        try:
            result = await self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行失败: {str(e)}"


class DuckDuckGoSearchTool(AgentTool):
    """DuckDuckGo联网搜索工具 - 免费无需API Key"""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        super().__init__(
            name="web_search",
            description="搜索互联网获取最新信息，支持天气、新闻、股价等实时查询",
            func=self._search
        )

    async def _search(self, query: str) -> str:
        """执行搜索"""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return "没有找到相关信息"

            response = "搜索结果:\n\n"
            for r in results[:3]:
                response += f"- {r.get('title', '')}\n"
                response += f"  {r.get('href', '')}\n"
                content = r.get('body', '')[:200]
                if content:
                    response += f"  {content}...\n\n"
            return response
        except Exception as e:
            return f"搜索失败: {str(e)}"


class GetCurrentTimeTool(AgentTool):
    """获取当前时间工具"""

    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="获取当前的日期和时间。用于需要知道现在是什么时候的问题，如'今天几号'、'现在几点'、'当前时间'等",
            func=self._get_time
        )

    async def _get_time(self, timezone: str = "Asia/Shanghai") -> str:
        """获取当前时间"""
        from datetime import datetime
        now = datetime.now()
        weekday_map = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weekday_map[now.weekday()]
        return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekday}"


# ── ReAct Agent (LangGraph) ────────────────────────────

class ReActAgent:
    """ReAct Agent - 基于 LangGraph StateGraph"""

    def __init__(self, llm_service, tools: List[AgentTool], config: Dict[str, Any]):
        self.llm_service = llm_service
        self.tools = {t.name: t for t in tools}
        self.config = config

        self.model_call_limit = config.get("model_call_limit", 8)
        self.tool_call_limit = config.get("tool_call_limit", 6)
        self.max_steps = config.get("max_steps", 10)

        self._graph = self._build_graph()

    # ── Graph construction ──────────────────────────────

    def _build_graph(self):
        """构建 LangGraph StateGraph"""
        builder = StateGraph(ReActState)

        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)

        builder.add_edge(START, "agent")

        builder.add_conditional_edges(
            "agent",
            self._router,
            {"tools": "tools", END: END}
        )

        builder.add_edge("tools", "agent")

        return builder.compile()

    # ── Nodes ───────────────────────────────────────────

    async def _agent_node(self, state: ReActState) -> Dict:
        """Agent 节点：构建 prompt、调用 LLM、解析 action"""
        prompt = self._build_react_prompt(
            state["question"],
            list(state.get("messages", [])),
        )

        response_text = await self.llm_service.chat(prompt)
        model_calls = state.get("model_calls", 0) + 1

        if not self.tools:
            return {
                "messages": [{"role": "assistant", "content": response_text}],
                "model_calls": model_calls,
                "final_answer": response_text,
            }

        action = self._parse_action(response_text)
        if action is None:
            return {
                "messages": [{"role": "assistant", "content": response_text}],
                "model_calls": model_calls,
                "final_answer": response_text,
            }

        return {
            "messages": [{"role": "assistant", "content": response_text}],
            "model_calls": model_calls,
            "pending_tool_name": action.get("tool", ""),
            "pending_tool_params": action.get("params", {}),
        }

    async def _tools_node(self, state: ReActState) -> Dict:
        """Tools 节点：执行工具并返回观察结果"""
        tool_name = state.get("pending_tool_name", "")
        tool_params = state.get("pending_tool_params", {})
        tool_calls = state.get("tool_calls", 0) + 1

        tool = self.tools.get(tool_name)
        step = len(state.get("trace_steps", [])) + 1

        if tool is None:
            observation = f"工具 {tool_name} 不存在"
            trace_entry = {
                "step": step,
                "tool": tool_name,
                "tool_params": tool_params,
                "tool_result": observation,
            }
            return {
                "messages": [{"role": "system", "content": observation}],
                "tool_calls": tool_calls,
                "used_tools": [tool_name],
                "trace_steps": [trace_entry],
                "final_answer": state["messages"][-1]["content"] if state.get("messages") else "",
            }

        try:
            tool_result = await tool.execute(**tool_params)
        except Exception as e:
            tool_result = f"工具执行失败: {str(e)}"

        observation = f"工具 {tool_name} 返回: {tool_result}"
        trace_entry = {
            "step": step,
            "tool": tool_name,
            "tool_params": tool_params,
            "tool_result": tool_result,
        }

        return {
            "messages": [
                {"role": "system", "content": observation},
            ],
            "tool_calls": tool_calls,
            "used_tools": [tool_name],
            "trace_steps": [trace_entry],
            "pending_tool_name": "",
            "pending_tool_params": {},
        }

    # ── Router ──────────────────────────────────────────

    def _router(self, state: ReActState) -> str:
        """条件路由：有工具调用且未超限 → tools，否则 → END"""
        final_answer = state.get("final_answer", "")
        if final_answer:
            return END

        # 错误状态
        if state.get("error"):
            return END

        model_calls = state.get("model_calls", 0)
        if model_calls >= self.model_call_limit:
            return END

        tool_calls = state.get("tool_calls", 0)
        if tool_calls >= self.tool_call_limit:
            return END

        pending_tool = state.get("pending_tool_name", "")
        if pending_tool and pending_tool in self.tools:
            return "tools"

        return END

    # ── Public API ──────────────────────────────────────

    async def execute(self, question: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行 ReAct 循环（非流式）"""
        history = history or []
        initial_messages = []
        for h in history:
            initial_messages.append(h)

        initial_state: ReActState = {
            "messages": initial_messages,
            "question": question,
            "model_calls": 0,
            "tool_calls": 0,
            "used_tools": [],
            "trace_steps": [],
            "pending_tool_name": "",
            "pending_tool_params": {},
            "final_answer": "",
            "error": "",
        }

        result = await self._graph.ainvoke(initial_state)

        return {
            "answer": result.get("final_answer", "") or self._extract_last_answer(result.get("messages", [])),
            "model_calls": result.get("model_calls", 0),
            "tool_calls": result.get("tool_calls", 0),
            "used_tools": list(result.get("used_tools", [])),
            "trace": {"steps": list(result.get("trace_steps", []))},
            "error": result.get("error") or None,
        }

    async def execute_stream(self, question: str, history: List[Dict[str, Any]] = None) -> AsyncIterator[str]:
        """执行 ReAct 循环（流式），逐 token yield

        注：LangGraph 0.2.0 的 astream() 只支持节点级事件，不支持 per-token 流式。
        因此流式路径使用手动循环，直接调用 llm_service.chat_stream() 实现 token 级流式。
        非流式路径 execute() 使用 LangGraph StateGraph。
        """
        history = history or []
        messages = list(history)
        model_calls = 0
        tool_calls = 0

        for step in range(self.max_steps):
            if model_calls >= self.model_call_limit:
                yield f"\n\n[已达到最大模型调用次数限制]"
                break

            prompt = self._build_react_prompt(question, messages)
            full_response = ""

            async for token in self.llm_service.chat_stream(prompt):
                full_response += token
                yield token

            model_calls += 1

            if not self.tools:
                break

            action = self._parse_action(full_response)
            if action is None:
                break

            tool_name = action.get("tool", "")
            tool_params = action.get("params", {})

            if tool_calls >= self.tool_call_limit:
                yield f"\n\n[已达到最大工具调用次数限制]"
                break

            tool = self.tools.get(tool_name)
            if tool is None:
                break

            try:
                tool_result = await tool.execute(**tool_params)
            except Exception as e:
                tool_result = f"工具执行失败: {str(e)}"

            tool_calls += 1

            observation = f"工具 {tool_name} 返回: {tool_result}"
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "system", "content": observation})

    # ── Prompt helpers ──────────────────────────────────

    def _build_react_prompt(self, query: str, history: List[Dict]) -> str:
        """构建ReAct提示词"""
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}" for t in self.tools.values()
        ])

        history_lines = []
        for h in history[-6:]:
            role = h.get("role", "")
            content = h.get("content", "")
            if role == "system":
                history_lines.append(f"观察: {content}")
            else:
                history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        return f"""你是一个AI助手，可以调用工具来回答用户问题。

可用工具:
{tools_desc}

对话历史:
{history_text}

用户: {query}

请按以下格式输出你的思考和动作：
思考: <你的思考>
动作: <工具名称>
参数: <JSON参数对象>

如果没有更多动作需要执行，直接回答。"""

    def _parse_action(self, response: str) -> Optional[Dict]:
        """解析动作"""
        lines = response.split("\n")
        tool = None
        params = {}

        for line in lines:
            line = line.strip()
            if line.startswith("动作:"):
                tool = line.replace("动作:", "").strip()
            elif line.startswith("参数:"):
                try:
                    params = json.loads(line.replace("参数:", "").strip())
                except (json.JSONDecodeError, ValueError):
                    params = {}

        if tool:
            return {"tool": tool, "params": params}
        return None

    def _extract_last_answer(self, messages: List[Dict]) -> str:
        """从消息列表中提取最后的回答"""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""


# ── Interceptors ───────────────────────────────────────

class RetryInterceptor:
    """工具重试拦截器 - """

    def __init__(self, max_retries: int = 2, initial_delay: float = 0.2, max_delay: float = 1.2):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """带重试的执行"""
        delay = self.initial_delay
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    import random
                    jitter = random.uniform(0, 0.1 * delay)
                    await asyncio.sleep(delay + jitter)
                    delay = min(delay * 2, self.max_delay)

        raise last_error


class ErrorInterceptor:
    """错误拦截器 -"""

    def __init__(self, fallback_value: str = "工具执行失败"):
        self.fallback_value = fallback_value

    async def execute_with_fallback(self, func: Callable, *args, **kwargs) -> str:
        """带错误兜底的执行"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return f"{self.fallback_value}: {str(e)}"
