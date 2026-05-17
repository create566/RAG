

@echo off
REM 初始化中间件数据脚本
REM 使用: init_data.bat

echo ========================================
echo   初始化 ES 索引和 Neo4j 图谱数据
echo ========================================

echo.
echo [1/2] 初始化 Elasticsearch 索引...
python scripts/init_elasticsearch.py --index super_agent_keywords --sample

echo.
echo [2/2] 初始化 Neo4j 图谱数据...
python scripts/init_neo4j_graph.py --demo

echo.
echo ========================================
echo   初始化完成！
echo ========================================
pause