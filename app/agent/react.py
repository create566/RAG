"""
LangGraph Agent核心模块
基于状态机实现ReAct Agent，对标Java的ReactAgentExecutor
"""
from typing import List, Dict, Any, Optional, Callable, TypedDict
from enum import Enum
import asyncio


class AgentState(TypedDict):
    """Agent状态"""
    messages: List[Dict[str, Any]]
    model_calls: int
    tool_calls: int
    used_tools: List[str]
    trace: Dict[str, Any]
    error: Optional[str]


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

            response = f"搜索结果:\n\n"
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


class ReActAgent:
    """ReAct Agent - 对标Java的ReactAgentExecutor"""

    def __init__(self, llm_service, tools: List[AgentTool], config: Dict[str, Any]):
        self.llm_service = llm_service
        self.tools = tools
        self.config = config

        self.model_call_limit = config.get("model_call_limit", 8)
        self.tool_call_limit = config.get("tool_call_limit", 6)
        self.max_steps = config.get("max_steps", 10)

    async def execute(self, question: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行ReAct循环"""
        history = history or []
        model_calls = 0
        tool_calls = 0
        used_tools = []
        trace_steps = []

        current_question = question
        observation = ""

        for step in range(self.max_steps):
            # 检查限制
            if model_calls >= self.model_call_limit:
                return {
                    "answer": "已达到最大模型调用次数限制",
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "used_tools": used_tools,
                    "trace": {"steps": trace_steps},
                    "error": "Model call limit exceeded"
                }

            # 构建ReAct提示词
            prompt = self._build_react_prompt(current_question, history, observation)

            # 调用模型
            response = await self.llm_service.chat(prompt)
            model_calls += 1

            # 如果没有可用工具，且模型输出了动作，检查是否可以用回答代替
            if not self.tools:
                # 没有工具，直接返回回答
                return {
                    "answer": response,
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "used_tools": used_tools,
                    "trace": {"steps": trace_steps}
                }

            # 解析动作
            action = self._parse_action(response)
            if action is None:
                # 没有动作，返回当前回答
                return {
                    "answer": response,
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "used_tools": used_tools,
                    "trace": {"steps": trace_steps}
                }

            tool_name = action.get("tool")
            tool_params = action.get("params", {})

            # 检查工具调用限制
            if tool_calls >= self.tool_call_limit:
                return {
                    "answer": "已达到最大工具调用次数限制",
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "used_tools": used_tools,
                    "trace": {"steps": trace_steps},
                    "error": "Tool call limit exceeded"
                }

            # 查找并执行工具
            tool_result = None
            for tool in self.tools:
                if tool.name == tool_name:
                    tool_result = await tool.execute(**tool_params)
                    tool_calls += 1
                    used_tools.append(tool_name)
                    break

            if tool_result is None:
                tool_result = f"工具 {tool_name} 不存在"

            # 如果工具不存在，直接返回回答（不再继续循环）
            if "不存在" in tool_result:
                return {
                    "answer": response,
                    "model_calls": model_calls,
                    "tool_calls": tool_calls,
                    "used_tools": used_tools,
                    "trace": {"steps": trace_steps}
                }

            trace_steps.append({
                "step": step + 1,
                "model_output": response,
                "tool": tool_name,
                "tool_params": tool_params,
                "tool_result": tool_result
            })

            observation = f"工具 {tool_name} 返回: {tool_result}"
            history.append({"role": "assistant", "content": response})
            history.append({"role": "system", "content": observation})

        return {
            "answer": observation,
            "model_calls": model_calls,
            "tool_calls": tool_calls,
            "used_tools": used_tools,
            "trace": {"steps": trace_steps}
        }

    def _build_react_prompt(self, query: str, history: List[Dict], observation: str) -> str:
        """构建ReAct提示词"""
        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])

        history_text = "\n".join([
            f"{h['role']}: {h['content']}"
            for h in history[-6:]
        ])

        return f"""你是一个AI助手，可以调用工具来回答用户问题。

可用工具:
{tools_desc}

对话历史:
{history_text}

{'上一步观察: ' + observation if observation else ''}

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
                    import json
                    params = json.loads(line.replace("参数:", "").strip())
                except:
                    params = {}

        if tool:
            return {"tool": tool, "params": params}
        return None


class RetryInterceptor:
    """工具重试拦截器 - 对标Java的ToolRetryInterceptor"""

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
                    # 指数退避 + 随机抖动
                    import random
                    jitter = random.uniform(0, 0.1 * delay)
                    await asyncio.sleep(delay + jitter)
                    delay = min(delay * 2, self.max_delay)

        raise last_error


class ErrorInterceptor:
    """错误拦截器 - 对标Java的ToolErrorInterceptor"""

    def __init__(self, fallback_value: str = "工具执行失败"):
        self.fallback_value = fallback_value

    async def execute_with_fallback(self, func: Callable, *args, **kwargs) -> str:
        """带错误兜底的执行"""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return f"{self.fallback_value}: {str(e)}"