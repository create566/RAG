"""
用户认证模型
"""
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class User(BaseModel):
    """用户模型"""
    id: Optional[int] = None
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    full_name: Optional[str] = None
    password_hash: str
    created_at: Optional[datetime] = None


class UserCreate(BaseModel):
    """用户注册请求"""
    username: str
    password: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """用户登录请求"""
    username: str
    password: str


class Token(BaseModel):
    """Token响应"""
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str


class TokenData(BaseModel):
    """Token解析后的数据"""
    user_id: Optional[int] = None
    username: Optional[str] = None