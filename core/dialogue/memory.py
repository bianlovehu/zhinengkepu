"""
对话记忆模块
"""
import logging
from typing import List, Dict, Any, Optional
from collections import deque

from config import get_settings

logger = logging.getLogger(__name__)


class DialogueMemory:
    """
    对话记忆

    管理对话历史，支持：
    - 滑动窗口记忆
    - 摘要记忆
    - 重要信息提取
    """

    def __init__(
        self,
        max_history: int = 20,
        summary_enabled: bool = True,
        summary_threshold: int = 10
    ):
        """
        Args:
            max_history: 最大历史消息数
            summary_enabled: 是否启用摘要
            summary_threshold: 触发摘要的消息数
        """
        self.max_history = max_history
        self.summary_enabled = summary_enabled
        self.summary_threshold = summary_threshold
        self.history: deque = deque(maxlen=max_history)
        self.summary: Optional[str] = None

    def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        添加记忆

        Args:
            role: 角色 (user/assistant)
            content: 内容
            metadata: 额外信息
        """
        entry = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.history.append(entry)

        # 检查是否需要摘要
        if self.summary_enabled and len(self.history) >= self.summary_threshold:
            self._maybe_summarize()

    def get_history(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        获取历史

        Args:
            last_n: 只返回最近N条

        Returns:
            List[Dict]: 消息列表
        """
        if last_n:
            return list(self.history)[-last_n:]
        return list(self.history)

    def get_context_for_llm(self) -> str:
        """
        获取适合LLM的上下文格式

        Returns:
            str: 格式化的上下文字符串
        """
        parts = []

        # 添加摘要（如果有）
        if self.summary:
            parts.append(f"【对话摘要】\n{self.summary}\n")

        # 添加最近历史
        recent = list(self.history)[-10:]  # 最近10条
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}: {msg['content']}")

        return "\n".join(parts) if parts else ""

    def _maybe_summarize(self):
        """检查并生成摘要"""
        if len(self.history) < self.summary_threshold:
            return

        # 简单摘要：提取关键信息
        user_msgs = [m["content"] for m in self.history if m["role"] == "user"]

        if len(user_msgs) >= self.summary_threshold:
            self.summary = self._generate_summary(user_msgs)
            logger.info("Generated conversation summary")

    def _generate_summary(self, messages: List[str]) -> str:
        """
        生成摘要

        简化实现，实际可用LLM
        """
        # 提取关键主题
        topics = set()
        for msg in messages:
            # 简单关键词提取
            keywords = ["型号", "故障", "维修", "物流", "尺寸", "更换", "退款"]
            for kw in keywords:
                if kw in msg:
                    topics.add(kw)

        if topics:
            return f"对话涉及主题: {', '.join(topics)}"
        return "对话涉及一般咨询"

    def clear(self):
        """清空记忆"""
        self.history.clear()
        self.summary = None

    def extract_entities(self) -> Dict[str, List[str]]:
        """
        提取实体信息

        如产品型号、订单号等
        """
        entities = {
            "product_models": [],
            "order_ids": [],
            "other": []
        }

        for msg in self.history:
            content = msg.get("content", "")

            # 提取型号
            import re
            models = re.findall(r'[A-Z]{2,}[0-9]{2,}[A-Z0-9]*', content)
            entities["product_models"].extend(models)

        # 去重
        entities["product_models"] = list(set(entities["product_models"]))

        return entities

    def get_important_info(self, key: str) -> Optional[str]:
        """
        获取重要信息

        如用户提到的产品型号等
        """
        entities = self.extract_entities()

        if key == "product_model" and entities["product_models"]:
            return entities["product_models"][0]

        return None
