"""
LLM服务 - 对标Java的ObservedChatModelService
支持多种大模型：DashScope、OpenAI、Anthropic
"""
from typing import Dict, Any, Optional, List, AsyncIterator
import asyncio
import os


from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseLLMService:
    """LLM服务基类"""

    async def chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        raise NotImplementedError

    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class DashScopeLLMService(BaseLLMService):
    """阿里云百炼LLM服务"""

    def __init__(self, api_key: str, model: str = "qwen-turbo", config: Dict = None):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.config = config or {}
        self.base_url = self.config.get("base_url", "https://dashscope.aliyuncs.com/api/v1")
        self._embedding_failure_count = 0
        self.embedding_available = True
        # 从环境变量读取 embedding 模型
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v1")
        logger.info(f"Using embedding model: {self.embedding_model}")

    async def chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """调用DashScope API"""
        try:
            import dashscope
            from dashscope import Generation

            dashscope.api_key = self.api_key

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            def _call():
                return Generation.call(
                    model=self.model,
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.8),
                    request_timeout=180,  # 180秒超时（qwen-max 响应较慢）
                )

            response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=190)

            if response.status_code == 200:
                output = response.output
                if output is None:
                    return ""
                return output.get("text", "")
            else:
                return f"API调用失败: {response.message}"
        except asyncio.TimeoutError:
            return "API调用超时（190秒）"
        except Exception as e:
            error_str = str(e)
            if "Connection aborted" in error_str or "RemoteDisconnected" in error_str or "ProxyError" in error_str:
                return "网络连接失败，请检查网络或代理设置"
            elif "timeout" in error_str.lower():
                return "请求超时，请稍后重试"
            else:
                return f"调用失败: {error_str}"

    async def chat_stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        """流式调用DashScope API - 逐token yield"""
        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.api_key

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream():
            try:
                responses = Generation.call(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    incremental_output=True,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.8),
                    request_timeout=180,  # 180秒超时（qwen-max 响应较慢）
                )
                for response in responses:
                    if response.status_code == 200:
                        if response.output:
                            text = response.output.get("text", "")
                            if text:
                                loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
                    else:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            ("error", f"API调用失败: {response.message}")
                        )
                        return
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                error_str = str(e)
                if "Connection aborted" in error_str or "RemoteDisconnected" in error_str or "ProxyError" in error_str:
                    error_str = "网络连接失败，请检查网络或代理设置"
                elif "timeout" in error_str.lower():
                    error_str = "请求超时，请稍后重试"
                loop.call_soon_threadsafe(queue.put_nowait, ("error", error_str))

        loop.run_in_executor(None, _stream)

        while True:
            event_type, data = await queue.get()
            if event_type == "done":
                break
            elif event_type == "error":
                yield data
                break
            else:
                yield data

    async def embed(self, text: str) -> List[float]:
        """获取文本嵌入（带重试）"""
        import dashscope
        from dashscope import TextEmbedding

        dashscope.api_key = self.api_key
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                def _call():
                    return TextEmbedding.call(
                        model=self.embedding_model,
                        input=text
                    )

                response = await asyncio.to_thread(_call)

                if response.status_code == 200:
                    embeddings = response.output
                    if embeddings is None:
                        return []
                    embeds = embeddings.get("embeddings", [{}])
                    if embeds and embeds[0].get("embedding"):
                        self._embedding_failure_count = 0  # 成功后重置计数器
                        self.embedding_available = True
                        logger.debug(f"[EMBED] 向量生成成功 | dim={len(embeds[0]['embedding'])}")
                        return embeds[0]["embedding"]
                    return []
                else:
                    logger.warning(f"Embedding API error (attempt {attempt+1}/{max_retries}): {response.message}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay * (2 ** attempt))
                    continue
            except Exception as e:
                logger.warning(f"Embedding failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                continue

        self._embedding_failure_count += 1
        if self._embedding_failure_count >= 3:
            self.embedding_available = False
            logger.warning(f"Embedding failed {self._embedding_failure_count} consecutive times, vector search disabled, keyword search only")
        logger.error(f"Embedding all {max_retries} retries failed")
        return []


class OpenAILLMService(BaseLLMService):
    """OpenAI LLM服务"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo", base_url: str = None, config: Dict = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.config = config or {}

    async def chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """调用OpenAI API"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            def _call():
                return client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7)
                )

            response = await asyncio.to_thread(_call)

            if response.choices:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            error_str = str(e)
            if "Connection aborted" in error_str or "RemoteDisconnected" in error_str or "ProxyError" in error_str:
                return "网络连接失败，请检查网络或代理设置"
            elif "timeout" in error_str.lower():
                return "请求超时，请稍后重试"
            else:
                return f"调用失败: {error_str}"

    async def embed(self, text: str) -> List[float]:
        """获取文本嵌入"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)

            def _call():
                return client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )

            response = await asyncio.to_thread(_call)

            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []


class AnthropicLLMService(BaseLLMService):
    """Anthropic LLM服务"""

    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229", config: Dict = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.config = config or {}

    async def chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """调用Anthropic API"""
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

            def _call():
                return client.messages.create(
                    model=self.model,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )

            response = await asyncio.to_thread(_call)

            return response.content[0].text
        except Exception as e:
            error_str = str(e)
            if "Connection aborted" in error_str or "RemoteDisconnected" in error_str or "ProxyError" in error_str:
                return "网络连接失败，请检查网络或代理设置"
            elif "timeout" in error_str.lower():
                return "请求超时，请稍后重试"
            else:
                return f"调用失败: {error_str}"

    async def embed(self, text: str) -> List[float]:
        # Anthropic不提供embedding服务，使用其他服务
        return []


class VLLMService(BaseLLMService):
    """VLLM 本地模型服务"""

    def __init__(self, model: str = "Qwen/Qwen3-1.8B-Instruct", base_url: str = "http://localhost:8010/v1", config: Dict = None):
        self.model = model
        self.base_url = base_url
        self.config = config or {}

    async def chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI(api_key="vllm", base_url=self.base_url)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call():
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.3),
            )

        response = await asyncio.to_thread(_call)
        if response.choices:
            return response.choices[0].message.content
        return ""

    async def chat_stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        from openai import OpenAI
        client = OpenAI(api_key="vllm", base_url=self.base_url)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream():
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.3),
                )
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                error_str = str(e)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", error_str))

        loop.run_in_executor(None, _stream)

        while True:
            event_type, data = await queue.get()
            if event_type == "done":
                break
            elif event_type == "error":
                yield data
                break
            else:
                yield data

    async def embed(self, text: str) -> List[float]:
        """获取文本嵌入 - VLLM embedding 服务"""
        try:
            from openai import OpenAI
            # embedding 使用单独的端口 8011
            embed_base_url = self.base_url.replace("8010", "8011")
            client = OpenAI(api_key="vllm", base_url=embed_base_url)
            # 从 config 读取 embedding 模型名，或 fallback 到 settings
            embed_model = self.config.get("embedding_model", "Alibaba-NLP/gte-multilingual-base")

            def _call():
                return client.embeddings.create(
                    model=embed_model,
                    input=text
                )

            response = await asyncio.to_thread(_call)
            if response.data:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"[VLLM] Embedding failed: {e}")
            return []


class OllamaLLMService(BaseLLMService):
    """Ollama 本地模型服务"""

    def __init__(self, model: str = "qwen2.5", base_url: str = "http://localhost:11434", config: Dict = None):
        self.model = model
        self.base_url = base_url
        self.config = config or {}

    async def chat(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=self.base_url)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _call():
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.7),
            )

        response = await asyncio.to_thread(_call)
        if response.choices:
            return response.choices[0].message.content
        return ""

    async def chat_stream(self, prompt: str, system_prompt: str = None, **kwargs) -> AsyncIterator[str]:
        from openai import OpenAI
        client = OpenAI(api_key="ollama", base_url=self.base_url)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream():
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    temperature=kwargs.get("temperature", 0.7),
                )
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                error_str = str(e)
                if "Connection aborted" in error_str or "RemoteDisconnected" in error_str or "ProxyError" in error_str:
                    error_str = "网络连接失败，请检查网络或代理设置"
                elif "timeout" in error_str.lower():
                    error_str = "请求超时，请稍后重试"
                loop.call_soon_threadsafe(queue.put_nowait, ("error", error_str))

        loop.run_in_executor(None, _stream)

        while True:
            event_type, data = await queue.get()
            if event_type == "done":
                break
            elif event_type == "error":
                yield data
                break
            else:
                yield data

    async def embed(self, text: str) -> List[float]:
        return []


def create_llm_service(provider: str = "dashscope", config: Dict = None) -> BaseLLMService:
    """工厂方法创建LLM服务"""
    config = config or {}

    if provider == "dashscope":
        return DashScopeLLMService(
            api_key=config.get("api_key", ""),
            model=config.get("model", "qwen-turbo"),
            config=config
        )
    elif provider == "openai":
        return OpenAILLMService(
            api_key=config.get("api_key", ""),
            model=config.get("model", "gpt-4-turbo"),
            base_url=config.get("base_url"),
            config=config
        )
    elif provider == "anthropic":
        return AnthropicLLMService(
            api_key=config.get("api_key", ""),
            model=config.get("model", "claude-3-opus-20240229"),
            config=config
        )
    elif provider == "ollama":
        return OllamaLLMService(
            model=config.get("model", "qwen2.5"),
            base_url=config.get("base_url", "http://localhost:11434"),
        )
    elif provider == "vllm":
        return VLLMService(
            model=config.get("model", "Qwen/Qwen3-1.8B-Instruct"),
            base_url=config.get("base_url", "http://localhost:8010/v1"),
            config=config,
        )
    else:
        raise ValueError(f"不支持的LLM provider: {provider}")