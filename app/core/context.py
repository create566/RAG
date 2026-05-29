"""
请求上下文管理 - 用 contextvars 存储当前请求的用户信息
"""
from contextvars import ContextVar
from typing import Optional

# 用户上下文
_current_user_id: ContextVar[Optional[int]] = ContextVar('current_user_id', default=None)
_current_username: ContextVar[Optional[str]] = ContextVar('current_username', default=None)


def set_user_context(user_id: int, username: str = None):
    """设置当前请求的用户上下文"""
    _current_user_id.set(user_id)
    _current_username.set(username)


def get_user_id() -> Optional[int]:
    """获取当前用户 ID"""
    return _current_user_id.get()


def get_username() -> Optional[str]:
    """获取当前用户名"""
    return _current_username.get()


def clear_user_context():
    """清除用户上下文"""
    _current_user_id.set(None)
    _current_username.set(None)


class UserContext:
    """用户上下文管理器，用于 with 语句"""

    def __init__(self, user_id: int, username: str = None):
        self.user_id = user_id
        self.username = username
        self._token = None

    def __enter__(self):
        self._token = set_user_context(self.user_id, self.username)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_user_context()
