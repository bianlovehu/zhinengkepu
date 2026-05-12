"""
LLM客户端基类
"""
import logging
from typing import List, Dict, Optional, Any

from config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM客户端封装

    支持多种LLM Provider: OpenAI, Zhipu, DashScope等
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._vision_client = None
        self.model = self.settings.LLM_MODEL
        self.vision_model = self.settings.VISION_MODEL

    @property
    def client(self):
        """延迟初始化LLM客户端"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def vision_client(self):
        """延迟初始化视觉模型客户端"""
        if self._vision_client is None:
            self._vision_client = self._create_vision_client()
        return self._vision_client

    def _create_client(self):
        """创建LLM客户端"""
        provider = self.settings.LLM_PROVIDER.lower()

        if provider == "openai":
            try:
                from openai import AsyncOpenAI
                return AsyncOpenAI(
                    api_key=self.settings.LLM_API_KEY,
                    base_url=self.settings.LLM_BASE_URL
                )
            except ImportError:
                logger.warning("OpenAI SDK not installed")

        elif provider == "zhipu":
            try:
                from zhipuai import AsyncZhipuAI
                return AsyncZhipuAI(api_key=self.settings.LLM_API_KEY)
            except ImportError:
                logger.warning("Zhipu SDK not installed")

        elif provider == "dashscope":
            try:
                import dashscope
                dashscope.api_key = self.settings.LLM_API_KEY
                return dashscope
            except ImportError:
                logger.warning("DashScope not installed")

        return None

    def _create_vision_client(self):
        """创建视觉模型客户端"""
        api_key = self.settings.VISION_API_KEY or self.settings.LLM_API_KEY
        base_url = self.settings.VISION_BASE_URL or self.settings.LLM_BASE_URL

        if not api_key:
            logger.warning("No API key configured for vision model")
            return None

        try:
            from openai import AsyncOpenAI
            return AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            logger.warning("OpenAI SDK not installed")
            return None

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_response: bool = False,
        model: Optional[str] = None
    ) -> str:
        """
        通用的聊天接口

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            json_response: 是否期望JSON响应
            model: 指定模型

        Returns:
            str: LLM响应文本
        """
        if self.client is None:
            logger.error("LLM client not initialized")
            return "LLM服务暂不可用"

        provider = self.settings.LLM_PROVIDER.lower()
        model = model or self.model
        temperature = temperature or self.settings.LLM_TEMPERATURE
        max_tokens = max_tokens or self.settings.LLM_MAX_TOKENS

        try:
            if provider == "openai":
                return await self._chat_openai(
                    messages, model, temperature, max_tokens, json_response
                )
            elif provider == "zhipu":
                return await self._chat_zhipu(
                    messages, model, temperature, max_tokens
                )
            elif provider == "dashscope":
                return await self._chat_dashscope(
                    messages, model, temperature, max_tokens
                )
            else:
                return "不支持的LLM Provider"

        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return f"LLM调用失败: {str(e)}"

    async def chat_with_vision(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        多模态聊天接口（支持图片）

        Args:
            messages: 消息列表，包含图片内容
            temperature: 温度参数
            max_tokens: 最大token数
            model: 指定视觉模型

        Returns:
            str: LLM响应文本
        """
        model = model or self.vision_model
        temperature = temperature or self.settings.LLM_TEMPERATURE
        max_tokens = max_tokens or self.settings.LLM_MAX_TOKENS

        # 使用视觉客户端
        client = self.vision_client or self.client
        if client is None:
            logger.error("No vision client available")
            return "视觉模型服务暂不可用"

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision chat error: {e}")
            return f"视觉模型调用失败: {str(e)}"

    async def _chat_openai(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int,
        json_response: bool
    ) -> str:
        """OpenAI API调用"""
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_response:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def _chat_zhipu(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """智谱GLM API调用"""
        response = await self.client.chat.completions.async_create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    async def _chat_dashscope(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """阿里通义千问API调用"""
        from dashscope import Generation
        response = Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.output.text
