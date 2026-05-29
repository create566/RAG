# Super Agent Python Backend - Dockerfile

FROM python:3.10

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 预装核心依赖（分批安装避免冲突）
RUN pip install --no-cache-dir \
    pydantic==2.6.4 \
    pydantic-settings==2.2.1 \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    httpx==0.27.0 \
    numpy==1.26.4 \
    loguru==0.7.2 \
    python-dotenv==1.0.1 \
    dashscope>=1.14.0 \
    kafka-python-ng==2.2.2 \
    sqlalchemy==2.0.30 \
    aiosqlite==0.20.0 \
    jieba==0.42.1 \
    pytest==8.1.1 \
    pytest-asyncio==0.23.6

# 安装 LangChain 相关依赖
RUN pip install --no-cache-dir \
    langchain==0.2.2 \
    langchain-core>=0.2.24 \
    langchain-openai==0.1.14 \
    langchain-anthropic==0.1.21 \
    langchain-community==0.2.2 \
    langgraph==0.0.62 \
    langchain-dashscope

# 安装其他依赖
RUN pip install --no-cache-dir \
    openai==1.14.0 \
    anthropic==0.25.0 \
    duckduckgo-search>=7.0.0 \
    elasticsearch==8.13.0 \
    neo4j==5.19.0 \
    tiktoken==0.7.0 \
    python-multipart==0.0.9 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4 \
    minio==7.2.3 \
    chromadb==0.5.0 \
    pymysql==1.1.1 \
    aiomysql==0.2.0 \
    cryptography==42.0.8

# 安装 ragas 评估相关
RUN pip install --no-cache-dir \
    ragas==0.3.0 \
    datasets \
    jsonplus

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p ./data/chroma

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "run.py"]