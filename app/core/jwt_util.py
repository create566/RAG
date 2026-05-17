"""
JWT工具类
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import jwt
import hashlib
import secrets

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码 - 使用盐值SHA256"""
    salt, stored_hash = hashed_password.split('$')
    pwd_hash = hashlib.sha256((salt + plain_password).encode()).hexdigest()
    return pwd_hash == stored_hash


def get_password_hash(password: str) -> str:
    """密码哈希 - 使用盐值SHA256（避免bcrypt兼容性问题）"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${pwd_hash}"


def create_access_token(user_id: int, username: str) -> str:
    """创建访问令牌"""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[Dict]:
    """验证令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None