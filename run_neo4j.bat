@echo off
REM Neo4j 快速启动脚本
REM 使用: run_neo4j.bat

echo ========================================
echo   Neo4j 图数据库快速启动
echo ========================================
echo.

REM 检查 Docker 是否安装
docker --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未安装 Docker，请先安装 Docker Desktop
    pause
    exit /b 1
)

REM 检查 Docker 是否运行
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] Docker 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)

echo [1/3] 启动 Neo4j 容器...
docker-compose up -d neo4j

echo.
echo [2/3] 等待 Neo4j 启动 (30秒)...
timeout /t 30 /nobreak >nul

echo.
echo [3/3] 初始化图谱数据...
python scripts/init_neo4j_graph.py --demo

echo.
echo ========================================
echo   Neo4j 启动完成！
echo   Web UI: http://localhost:7474
echo   Bolt: bolt://localhost:7687
echo ========================================
echo.
echo   初始用户名: neo4j
echo   初始密码: password
echo.
pause