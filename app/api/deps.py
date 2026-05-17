"""
FastAPI 依赖注入 — get_db, get_settings, get_current_user
"""
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import get_settings
from app.core.jwt_util import verify_token


async def get_settings_dep():
    """依赖: 应用配置"""
    return get_settings()


async def get_current_user(authorization: str = Header(None)):
    """依赖: 从 Bearer token 解析当前用户"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    token = authorization.replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return {
        "user_id": payload.get("user_id"),
        "username": payload.get("username"),
    }
