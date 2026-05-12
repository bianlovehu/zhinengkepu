"""
图片理解模块
"""
import base64
import json
import logging
from typing import Dict, Optional

from config import prompts
from core.llm.base import LLMClient

logger = logging.getLogger(__name__)


class ImageUnderstanding:
    """
    图片理解器

    使用多模态LLM理解用户上传的图片内容
    """

    def __init__(self):
        self.llm_client = LLMClient()

    async def describe(self, image_bytes: bytes) -> Dict:
        """
        描述图片内容

        Args:
            image_bytes: 图片二进制数据

        Returns:
            {
                "description": str,  # 图片描述
                "keywords": List[str],  # 关键词
                "relevance": str  # 与问题相关性
            }
        """
        try:
            # 将图片转为base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # 构建多模态消息
            prompt = prompts.IMAGE_DESCRIPTION_PROMPT

            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }

            # 调用多模态LLM
            response = await self.llm_client.chat_with_vision(
                messages=[message],
                temperature=0.3,
                model=self.llm_client.vision_model
            )

            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Failed to parse image description as JSON")
                return {
                    "description": response[:200] if response else "图片描述生成失败",
                    "keywords": [],
                    "relevance": "无法确定"
                }
            logger.info(f"Image description: {result.get('description', '')[:100]}")

            return result

        except json.JSONDecodeError:
            logger.warning("Failed to parse image description as JSON")
            return {
                "description": "图片内容解析失败",
                "keywords": [],
                "relevance": "无法确定"
            }
        except Exception as e:
            logger.error(f"Image understanding error: {e}")
            return {
                "description": f"图片理解出错: {str(e)}",
                "keywords": [],
                "relevance": "无法确定"
            }

    async def extract_text(self, image_bytes: bytes) -> str:
        """
        从图片中提取文字（OCR功能）

        适用于产品标签、说明书截图等
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请提取图片中的所有文字内容，保持原有格式。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }

            response = await self.llm_client.chat_with_vision(
                messages=[message],
                temperature=0.1,
                model=self.llm_client.vision_model
            )

            return response

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
