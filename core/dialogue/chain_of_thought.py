"""
思维链拆解模块
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

from config import prompts
from core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class ChainOfThought:
    """
    思维链拆解

    识别复杂问题并进行拆解，确保每个子问题都被回答
    """

    def __init__(self):
        self.llm_client = LLMClient()

    def should_decompose(self, question: str) -> Tuple[bool, List[str]]:
        """
        判断问题是否需要拆解

        Args:
            question: 用户问题

        Returns:
            Tuple[bool, List[str]]: (是否拆解, 子问题列表)
        """
        # 规则1：多个问号
        question_marks = question.count('？') + question.count('?')
        if question_marks > 1:
            return True, self._split_by_question_mark(question)

        # 规则2：特定连接词
        connectors = ['并且', '而且', '还有', '另外', '以及', '同时', '并且', '并且']
        for conn in connectors:
            if conn in question:
                return True, self._split_by_connector(question, conn)

        # 规则3：复合句式
        compound_patterns = [
            (r'(\w+)，(\w+)，', 2),  # 逗号分隔的多个事项
            (r'除了(.+?)还(.+?)', 2),  # 除了...还...
        ]
        for pattern, count in compound_patterns:
            matches = re.findall(pattern, question)
            if len(matches) >= count:
                return True, self._decompose_compound(question)

        return False, [question]

    def _split_by_question_mark(self, question: str) -> List[str]:
        """按问号拆分"""
        parts = re.split(r'[？?]', question)
        return [p.strip() for p in parts if p.strip()]

    def _split_by_connector(self, question: str, connector: str) -> List[str]:
        """按连接词拆分"""
        parts = question.split(connector)
        return [p.strip() for p in parts if p.strip()]

    def _decompose_compound(self, question: str) -> List[str]:
        """拆解复合句"""
        # 简化实现
        sentences = re.split(r'[，,]', question)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    async def decompose_with_llm(
        self,
        question: str
    ) -> Tuple[bool, List[Dict[str, str]]]:
        """
        使用LLM智能拆解问题

        Args:
            question: 用户问题

        Returns:
            Tuple[bool, List[Dict]]: (是否拆解, [{sub_q: str, focus: str}])
        """
        try:
            prompt = f"""请分析以下用户问题，判断是否包含多个子问题，并进行拆解。

问题：{question}

请返回JSON格式：
{{
    "needs_decompose": true/false,
    "sub_questions": [
        {{"question": "子问题1", "focus": "关注点1"}},
        {{"question": "子问题2", "focus": "关注点2"}}
    ]
}}
"""

            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_response=True
            )

            import json
            result = json.loads(response)

            needs_decompose = result.get("needs_decompose", False)
            sub_questions = result.get("sub_questions", [])

            return needs_decompose, sub_questions

        except Exception as e:
            logger.error(f"LLM decompose error: {e}")
            return False, [{"question": question, "focus": "通用"}]

    def decompose_for_answer(
        self,
        question: str,
        search_results: Dict[str, Any]
    ) -> str:
        """
        拆解问题并生成结构化答案

        Args:
            question: 用户问题
            search_results: 检索结果

        Returns:
            str: 结构化答案
        """
        should_decompose, sub_questions = self.should_decompose(question)

        if not should_decompose:
            return None  # 不需要拆解

        # 构建答案框架
        answer_parts = []

        if len(sub_questions) > 1:
            for i, sub_q in enumerate(sub_questions, 1):
                answer_parts.append(f"【问题{i}】{sub_q}")
                # 实际答案需要根据检索结果填充
                answer_parts.append(f"（根据检索结果填充答案）\n")

        return "\n".join(answer_parts)

    def get_answer_structure(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        获取答案结构

        Args:
            question: 用户问题

        Returns:
            Dict: {
                "type": "single/composite",
                "parts": [{"question": str, "answer_placeholder": str}]
            }
        """
        should_decompose, sub_questions = self.should_decompose(question)

        if should_decompose:
            return {
                "type": "composite",
                "parts": [{"question": q, "answer": ""} for q in sub_questions]
            }
        else:
            return {
                "type": "single",
                "parts": [{"question": question, "answer": ""}]
            }
