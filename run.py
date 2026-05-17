#!/usr/bin/env python3
"""
Super Agent Python Backend 启动脚本
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 环境变量
from dotenv import load_dotenv

load_dotenv('.env')

if __name__ == "__main__":
    import uvicorn
    from app.config import get_settings

    settings = get_settings()

    print("""
╔═══════════════════════════════════════════════════════════╗
║           Super Agent Python Backend                      ║
║           企业级AI智能体对话平台                          ║
║           版本: 1.0.0                                     ║
╠═══════════════════════════════════════════════════════════╣
║  服务器地址: http://{host}:{port}                          ║
║  API文档:    http://{host}:{port}/docs                     ║
╚═══════════════════════════════════════════════════════════╝
    """.format(host=settings.server_host, port=settings.server_port))

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload
    )
