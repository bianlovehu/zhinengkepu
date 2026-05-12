"""
认证中间件
"""
from fastapi import HTTPException, Header
from config import get_settings


def verify_token(authorization: str) -> bool:
    """
    验证Bearer Token认证

    Args:
        authorization: Authorization header值

    Returns:
        bool: 验证是否通过

    Raises:
        HTTPException: 认证失败时抛出
    """
    settings = get_settings()

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Expected 'Bearer {token}'"
        )

    token = authorization[7:]  # 去掉 "Bearer " 前缀

    if token != settings.API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    return True
