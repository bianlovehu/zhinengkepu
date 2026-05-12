"""
API响应模型
"""
from typing import Optional
from pydantic import BaseModel


class ResponseData(BaseModel):
    """响应数据体"""
    answer: str = ...
    session_id: str = ...
    timestamp: int = ...


class ChatResponse(BaseModel):
    """聊天响应模型"""
    code: int = 0
    msg: str = "success"
    data: ResponseData = ...


class ErrorResponse(BaseModel):
    """错误响应模型"""
    code: int = 1
    msg: str
    detail: Optional[str] = None
