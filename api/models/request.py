"""
API请求模型
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型"""

    question: str = Field(
        ...,
        min_length=1,
        description="用户问题，客服场景的核心输入"
    )

    images: Optional[List[str]] = Field(
        default=None,
        description="Base64编码的图片列表，支持0-3张，每张≤5MB"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="会话ID，用于多轮对话"
    )

    stream: Optional[bool] = Field(
        default=False,
        description="是否流式响应"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "我的DCB107电钻指示灯闪烁是什么意思？",
                "images": ["data:image/png;base64,iVBORw0KGgo..."],
                "session_id": "kf_session_889900"
            }
        }
