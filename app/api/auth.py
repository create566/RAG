"""
用户认证 API
"""
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from app.models.user import UserCreate, UserLogin, Token, User
from app.core.jwt_util import verify_password, get_password_hash, create_access_token, verify_token
from app.config import get_settings
import pymysql
from pymysql.cursors import DictCursor

router = APIRouter(prefix="/api/auth", tags=["认证"])


def get_db_connection():
    """从配置获取数据库连接"""
    s = get_settings()
    return pymysql.connect(
        host=s.mysql.host,
        port=s.mysql.port,
        user=s.mysql.username,
        password=s.mysql.password,
        database=s.mysql.database,
        charset="utf8mb4",
        cursorclass=DictCursor,
    )


@router.post("/register", response_model=Token)
async def register(user: UserCreate):
    """注册新用户"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 检查用户名是否已存在
            cursor.execute("SELECT id FROM users WHERE username = %s", (user.username,))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="用户名已存在")

            # 密码哈希
            password_hash = get_password_hash(user.password)

            # 插入用户
            cursor.execute(
                "INSERT INTO users (username, email, phone, department, position, full_name, password_hash) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user.username, user.email, user.phone, user.department, user.position, user.full_name, password_hash)
            )
            conn.commit()

            # 获取用户ID
            cursor.execute("SELECT id FROM users WHERE username = %s", (user.username,))
            db_user = cursor.fetchone()
            user_id = db_user["id"]

            # 生成token
            access_token = create_access_token(user_id, user.username)
            return Token(
                access_token=access_token,
                token_type="bearer",
                user_id=user_id,
                username=user.username
            )
    finally:
        conn.close()


@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    """用户登录"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 查找用户
            cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (user.username,))
            db_user = cursor.fetchone()

            if not db_user:
                raise HTTPException(status_code=401, detail="用户名或密码错误")

            # 验证密码
            if not verify_password(user.password, db_user["password_hash"]):
                raise HTTPException(status_code=401, detail="用户名或密码错误")

            # 生成token
            access_token = create_access_token(db_user["id"], db_user["username"])
            return Token(
                access_token=access_token,
                token_type="bearer",
                user_id=db_user["id"],
                username=db_user["username"]
            )
    finally:
        conn.close()


@router.get("/me", response_model=User)
async def get_current_user(authorization: str = Header(...)):
    """获取当前用户信息"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="无效的认证头")

    token = authorization.replace("Bearer ", "")
    payload = verify_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="无效或过期的token")

    user_id = payload.get("user_id")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = %s", (user_id,))
            db_user = cursor.fetchone()

            if not db_user:
                raise HTTPException(status_code=404, detail="用户不存在")

            return User(
                id=db_user["id"],
                username=db_user["username"],
                email=db_user.get("email"),
                password_hash="",
                created_at=db_user["created_at"]
            )
    finally:
        conn.close()