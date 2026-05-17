"""
FastAPI应用入口 - 对标Java的SuperBusinessChatAgentApplication
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import get_settings
from app.api import chat_router, document_router, auth_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("Super Agent Python Backend 启动中...")
    yield
    print("Super Agent Python Backend 关闭中...")


app = FastAPI(
    title="Super Agent",
    description="企业级AI智能体对话平台 Python版",
    version="1.0.0",
    lifespan=lifespan
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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
        reload=settings.server_reload
    )