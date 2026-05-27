"""
异步数据库连接池 - SQLAlchemy async + aiomysql
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import get_settings

_async_engine = None
_async_session_factory = None


def _get_engine():
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.mysql.async_url,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            echo=False,
        )
    return _async_engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_db():
    """FastAPI 依赖: 获取数据库会话"""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_async_session():
    """独立事务用的异步会话（不依赖 FastAPI 依赖注入）"""
    factory = get_session_factory()
    async with factory() as session:
        return session


async def close_db():
    """关闭连接池"""
    global _async_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
