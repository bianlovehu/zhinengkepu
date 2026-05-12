"""
API模型初始化
"""
from .request import ChatRequest
from .response import ChatResponse, ResponseData, ErrorResponse

__all__ = ["ChatRequest", "ChatResponse", "ResponseData", "ErrorResponse"]
