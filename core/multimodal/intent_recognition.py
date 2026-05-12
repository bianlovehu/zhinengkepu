"""
意图识别模块
"""
import json
import logging
from typing import Dict, List, Optional

from config import get_settings, prompts
from core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class IntentRecognizer:
    """
    意图识别器

    分析用户问题，识别：
    - 问题意图类型
    - 关键词
    - 涉及的产品
    - 是否需要图片辅助
    """

    def __init__(self):
        self.llm_client = LLMClient()
        self.settings = get_settings()

    async def recognize(self, question: str, images: Optional[List[bytes]] = None) -> Dict:
        """
        识别用户意图

        Args:
            question: 用户问题
            images: 可选的图片列表

        Returns:
            Dict: {
                "intent": str,  # 意图类型
                "keywords": List[str],  # 关键词
                "product_mentioned": str,  # 提及的产品
                "needs_images": bool  # 是否需要图片
            }
        """
        try:
            # 如果有图片，先理解图片内容
            image_context = ""
            if images:
                from core.multimodal.image_understanding import ImageUnderstanding
                img_understood = ImageUnderstanding()
                for i, img_bytes in enumerate(images):
                    desc = await img_understood.describe(img_bytes)
                    image_context += f"\n图片{i+1}内容: {desc.get('description', '')}"

            # 构建提示词
            prompt = prompts.INTENT_RECOGNITION_PROMPT.format(question=question)

            if image_context:
                prompt += f"\n\n用户上传的图片信息:\n{image_context}"

            # 调用LLM识别意图
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_response=True
            )

            # 解析结果（处理可能的额外文本）
            result = self._parse_json_response(response)

            # 兼容旧格式：将 keywords 列表转换为新格式
            if isinstance(result.get("keywords"), list):
                keywords_dict = {
                    "high": result["keywords"][:2] if len(result["keywords"]) > 2 else result["keywords"],
                    "medium": [],
                    "low": []
                }
                result["keywords"] = keywords_dict

            logger.info(f"Intent recognition result: {result}")

            return result

        except Exception as e:
            logger.error(f"Intent recognition error: {e}")
            return self._fallback_recognition(question)

    def _parse_json_response(self, response: str) -> Dict:
        """解析 LLM 返回的 JSON 响应，处理可能的额外文本"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 尝试提取代码块中的 JSON
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON: {response[:100]}...")
        raise ValueError("Cannot parse JSON response")

    def _fallback_recognition(self, question: str) -> Dict:
        """意图识别失败时的兜底方案"""
        question_lower = question.lower()

        intents = []
        if any(kw in question_lower for kw in ["怎么", "如何", "方法", "步骤"]):
            intents.append("使用指导")
        if any(kw in question_lower for kw in ["坏了", "故障", "问题", "维修", "闪烁", "异常", "不工作"]):
            intents.append("故障咨询")
        if any(kw in question_lower for kw in ["送", "物流", "快递", "发货"]):
            intents.append("物流查询")
        if any(kw in question_lower for kw in ["更换", "退货", "退款", "售后"]):
            intents.append("售后服务")
        if any(kw in question_lower for kw in ["尺寸", "规格", "型号"]):
            intents.append("产品咨询")

        if not intents:
            intents = ["一般咨询"]

        # 判断是否需要图片辅助
        needs_images = any(kw in question_lower for kw in [
            "指示灯", "灯", "图", "显示", "屏幕", "按钮", "接口",
            "外观", "尺寸", "规格", "实物", "产品", "部件",
            "坏了", "故障", "维修", "怎么", "如何", "闪烁"
        ])

        return {
            "intent": "/".join(intents),
            "keywords": [],
            "product_mentioned": "",
            "needs_images": needs_images
        }
