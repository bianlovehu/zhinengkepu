"""
答案生成器
"""
import logging
from typing import List, Dict, Any, Optional

from config import prompts
from core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    答案生成器

    基于RAG检索结果和对话历史，生成最终答案
    """

    def __init__(self):
        self.llm_client = LLMClient()

    async def generate(
        self,
        question: str,
        context: Dict[str, Any],
        history: List[Dict[str, str]] = None,
        images: List[bytes] = None,
        intent: Dict[str, Any] = None
    ) -> str:
        """
        生成答案

        Args:
            question: 用户问题
            context: RAG检索结果，包含texts和images
            history: 对话历史
            images: 用户上传的图片
            intent: 意图识别结果

        Returns:
            str: 生成的答案
        """
        history = history or []

        # 构建上下文
        texts = context.get("texts", [])
        relevant_images = context.get("images", [])

        # 格式化文本上下文
        context_text = self._format_text_context(texts)

        # 格式化图片上下文
        context_images = self._format_image_context(relevant_images)

        # 构建系统提示词
        system_prompt = prompts.SYSTEM_PROMPT

        # 添加历史对话上下文
        history_context = self._format_history(history)
        if history_context:
            system_prompt += f"\n\n【对话历史】\n{history_context}"

        # 构建用户消息
        user_message = prompts.RAG_ANSWER_PROMPT.format(
            context=context_text,
            images=context_images,
            question=question
        )

        # 如果有用户上传的图片，添加到消息中
        if images:
            user_message = await self._add_user_images(user_message, images)

        # 调用LLM生成
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        logger.info(f"Generating answer for question: {question[:50]}...")

        answer = await self.llm_client.chat(
            messages=messages,
            temperature=0.7,
            json_response=False
        )

        # 后处理：添加图片标记
        answer = self._post_process(answer, relevant_images)

        return answer

    def _format_text_context(self, texts: List[Dict]) -> str:
        """格式化文本上下文"""
        if not texts:
            return "（知识库中未找到相关内容）"

        formatted = []
        for i, item in enumerate(texts, 1):
            source = item.get("source", "未知来源")
            content = item.get("content", "")
            score = item.get("score", 0)
            formatted.append(f"【文本{i}】(来源: {source}, 相关度: {score:.2f})\n{content}")

        return "\n\n".join(formatted)

    def _format_image_context(self, images: List[Dict]) -> str:
        """格式化图片上下文"""
        if not images:
            return "（无相关图片）"

        formatted = []
        for i, img in enumerate(images, 1):
            img_id = img.get("id", f"image_{i}")
            desc = img.get("description", "无描述")
            formatted.append(f"【图片{i}】ID: {img_id}\n描述: {desc}")

        return "\n\n".join(formatted)

    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        if not history:
            return ""

        formatted = []
        for msg in history[-5:]:  # 只取最近5轮
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            formatted.append(f"{role}: {content[:100]}...")

        return "\n".join(formatted)

    async def _add_user_images(self, text: str, images: List[bytes]) -> str:
        """添加用户上传的图片到消息"""
        # 实际实现中，这里需要构建多模态消息
        # 简化处理：添加图片数量说明
        return f"{text}\n\n（用户上传了{len(images)}张图片供参考）"

    def _post_process(self, answer: str, images: List[Dict]) -> str:
        """后处理答案"""
        # 确保答案不为空
        if not answer or not answer.strip():
            return "抱歉，我暂时无法回答这个问题，请稍后再试。"

        # 清理多余空白
        answer = "\n".join(line.strip() for line in answer.split("\n"))

        return answer
