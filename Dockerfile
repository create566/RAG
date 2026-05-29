# Super Agent Python Backend - Dockerfile

FROM python:3.10

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 创建数据目录
RUN mkdir -p ./data/chroma ./data/uploads

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8001

# 启动命令
CMD ["python", "run.py"]