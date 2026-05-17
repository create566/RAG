"""
JWT工具类 — 密钥和算法从统一配置读取
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import jwt
import hashlib
import secrets


def _get_config():
    from app.config import get_settings
    s = get_settings()
    return s.jwt_secret_key, s.jwt_algorithm, s.jwt_expire_hours


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码 - 使用盐值SHA256"""
    salt, stored_hash = hashed_password.split('$')
    pwd_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
    return pwd_hash == stored_hash


def get_password_hash(password: str) -> str:
    """密码哈希 - 使用盐值 SHA256"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def create_access_token(user_id: int, username: str) -> str:
    """创建访问令牌"""
    secret_key, algorithm, expire_hours = _get_config()
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=expire_hours)
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def verify_token(token: str) -> Optional[Dict]:
    """验证令牌"""
    try:
        secret_key, algorithm, _ = _get_config()
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except Exception:
        return None
