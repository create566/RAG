"""
FastAPI 应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api import chat_router, document_router, auth_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Super Agent Python Backend 启动中...")
    yield
    print("Super Agent Python Backend 关闭中...")


app = FastAPI(
    title="Super Agent",
    description="企业级 AI 智能体对话平台 Python 版",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — 从配置读取白名单
allowed_origins = [
    o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
app.include_router(document_router)
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Super Agent Python Backend", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
    )
