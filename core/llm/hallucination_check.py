"""
幻觉检测模块
"""
import json
import logging
from typing import Dict, List, Any

from config import prompts
from core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class HallucinationChecker:
    """
    幻觉检测器

    检查生成的回答是否与知识库内容一致，识别可能的幻觉
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.enabled = True

    async def check(self, answer: str, context: Dict[str, Any]) -> bool:
        """
        检查答案是否存在幻觉

        Args:
            answer: 待检查的答案
            context: 知识库上下文

        Returns:
            bool: True表示答案可信，False表示可能存在幻觉
        """
        if not self.enabled:
            return True

        try:
            # 提取上下文文本
            texts = context.get("texts", [])
            if not texts:
                logger.warning("No context available for hallucination check")
                return True

            context_text = "\n".join(t.get("content", "") for t in texts[:3])

            # 构建检测提示词
            prompt = prompts.HALLUCINATION_CHECK_PROMPT.format(
                context=context_text,
                answer=answer
            )

            # 调用LLM检测
            messages = [
                {"role": "user", "content": prompt}
            ]

            response = await self.llm_client.chat(
                messages=messages,
                temperature=0.1,
                json_response=True
            )

            # 解析结果
            result = json.loads(response)

            is_valid = result.get("is_valid", result.get("valid", True))

            if not is_valid:
                issues = result.get("issues", [])
                logger.warning(f"Potential hallucination detected: {issues}")

            return is_valid

        except json.JSONDecodeError:
            logger.warning("Failed to parse hallucination check result")
            return True  # 解析失败时默认通过
        except Exception as e:
            logger.error(f"Hallucination check error: {e}")
            return True  # 出错时默认通过，避免阻断服务

    async def batch_check(
        self,
        answers: List[str],
        contexts: List[Dict[str, Any]]
    ) -> List[bool]:
        """
        批量幻觉检测

        Args:
            answers: 答案列表
            contexts: 上下文列表

        Returns:
            List[bool]: 每个答案的检测结果
        """
        results = []
        for answer, context in zip(answers, contexts):
            result = await self.check(answer, context)
            results.append(result)

        return results

    def get_confidence(self, answer: str, context: Dict[str, Any]) -> float:
        """
        估算答案的置信度

        基于简单规则估算，用于辅助判断
        """
        confidence = 1.0

        # 检查答案长度
        if len(answer) < 10:
            confidence *= 0.5
        elif len(answer) > 5000:
            confidence *= 0.8

        # 检查是否包含"不确定"、"可能"等模糊词
        uncertain_words = ["不确定", "可能", "也许", "大概", "不清楚"]
        for word in uncertain_words:
            if word in answer:
                confidence *= 0.9

        # 检查是否有引用
        if "<PIC>" in answer or "来源" in answer or "根据" in answer:
            confidence *= 1.1

        return min(max(confidence, 0.0), 1.0)
