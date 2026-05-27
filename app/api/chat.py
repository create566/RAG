"""
聊天 API 接口 — 瘦控制器
"""
import json
from datetime import datetime, timedelta
from sqlalchemy import text

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.chat import ChatRequest, ChatResponse, SourceReference
from app.services.chat_service import get_chat_service
from app.config import get_settings
from app.core.database import get_async_session
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    service = get_chat_service()
    return await service.chat(request)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口 — 逐 token SSE 推送"""
    service = get_chat_service()
    conv_id = request.conversation_id or f"conv_{hash(request.question)}_{id(request)}"
    request.conversation_id = conv_id

    async def generate():
        try:
            async for token in service.chat_stream(request):
                yield f"data: {json.dumps({'answer': token, 'done': False}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'answer': '', 'done': True, 'conversation_id': conv_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            error_msg = f"错误: {e}"
            traceback.print_exc()
            yield f"data: {json.dumps({'answer': error_msg, 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """获取会话历史"""
    service = get_chat_service()
    memory_context = await service.memory_service.load_memory_context(conversation_id)
    return {"conversation_id": conversation_id, "context": memory_context}


@router.get("/conversations")
async def list_conversations(user_id: int = None):
    """列出用户的会话列表"""
    if not user_id:
        return {"success": False, "error": "需要 user_id 参数"}

    try:
        session = await get_async_session()
        async with session:
            # 确保表存在
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS conversation (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id INT NOT NULL DEFAULT 0,
                    title VARCHAR(255) DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    summary TEXT,
                    INDEX idx_user_id (user_id),
                    INDEX idx_updated_at (updated_at),
                    INDEX idx_user_updated (user_id, updated_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """))
            await session.execute(text("""
                CREATE TABLE IF NOT EXISTS message (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id VARCHAR(64) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_conversation_id (conversation_id),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """))
            await session.commit()

            # 查询会话列表
            result = await session.execute(
                text("SELECT id, user_id, title, summary, created_at, updated_at "
                     "FROM conversation WHERE user_id = :user_id "
                     "ORDER BY updated_at DESC LIMIT 50"),
                {"user_id": user_id}
            )
            convs = result.fetchall()

            for conv in convs:
                if not conv.title:
                    msg_result = await session.execute(
                        text("SELECT content FROM message WHERE conversation_id = :conv_id AND role = 'user' ORDER BY id ASC LIMIT 1"),
                        {"conv_id": conv.id}
                    )
                    first_msg = msg_result.fetchone()
                    if first_msg:
                        conv.title = first_msg.content[:50]

            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            date_groups = {}

            for conv in convs:
                conv_date = conv.updated_at.date() if conv.updated_at else today
                if conv_date == today:
                    date_label = "今天"
                elif conv_date == yesterday:
                    date_label = "昨天"
                else:
                    date_label = conv_date.strftime("%Y-%m-%d")

                if date_label not in date_groups:
                    date_groups[date_label] = []
                date_groups[date_label].append({
                    "id": conv.id,
                    "title": conv.title or conv.summary or "新对话",
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                })

            result_list = [{"date": k, "conversations": v} for k, v in date_groups.items()]
            return {"success": True, "date_groups": result_list}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """获取会话的全部消息"""
    try:
        session = await get_async_session()
        async with session:
            result = await session.execute(
                text("SELECT id, role, content, created_at FROM message WHERE conversation_id = :conv_id ORDER BY id ASC"),
                {"conv_id": conversation_id}
            )
            messages = [
                {
                    "id": m.id, "role": m.role, "content": m.content,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in result.fetchall()
            ]
            return {"success": True, "conversation_id": conversation_id, "messages": messages}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除会话"""
    try:
        session = await get_async_session()
        async with session:
            await session.execute(
                text("DELETE FROM message WHERE conversation_id = :conv_id"),
                {"conv_id": conversation_id}
            )
            await session.execute(
                text("DELETE FROM conversation WHERE id = :conv_id"),
                {"conv_id": conversation_id}
            )
            await session.commit()
            return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 图谱测试接口 ──────────────────────────────────────────

@router.get("/graph/document/{document_id}")
async def query_graph_document(document_id: int):
    """查询文档的图谱结构"""
    service = get_chat_service()
    try:
        if not service.graph_query_engine:
            return {"success": False, "error": "图查询引擎未初始化"}
        result = await service.graph_query_engine.query_document_structure(document_id)
        return {"success": True, "document_id": document_id, "structure": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/graph/document/{document_id}/chapter")
async def query_graph_chapter(document_id: int, section_hint: str = None):
    """查询文档特定章节内容"""
    service = get_chat_service()
    try:
        if not service.graph_query_engine:
            return {"success": False, "error": "图查询引擎未初始化"}

        if section_hint:
            result = await service.graph_query_engine.query_section_content(document_id, section_hint)
        else:
            result = await service.graph_query_engine.query_document_structure(document_id)

        return {"success": True, "document_id": document_id, "section_hint": section_hint, "result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/graph/cypher")
async def execute_cypher(cypher: str = None, query: str = None):
    """执行自定义 Cypher 查询"""
    from app.core.neo4j_client import Neo4jClient

    service = get_chat_service()
    settings = get_settings()
    try:
        if not service.graph_query_engine:
            return {"success": False, "error": "图查询引擎未初始化"}

        cypher_query = query or cypher
        if not cypher_query:
            return {"success": False, "error": "请提供 cypher 或 query 参数"}

        client = Neo4jClient(
            uri=settings.neo4j.uri,
            username=settings.neo4j.username,
            password=settings.neo4j.password,
        )

        if not client.health_check():
            return {"success": False, "error": "Neo4j 连接失败"}

        results = client.query(cypher_query)
        return {"success": True, "cypher": cypher_query, "count": len(results), "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
