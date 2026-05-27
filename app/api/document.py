"""
文档管理 API 接口 — 瘦控制器
"""
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form

from app.services.document_service import get_document_service
from app.api.deps import get_current_user
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/document", tags=["document"])

ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".xlsx", ".xls"}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = None,
    chunk_strategy: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user),
):
    """上传文档（同步处理），可选指定切块策略"""
    user_id = current_user.get("user_id")
    logger.info(f"UPLOAD: file={file.filename}, user_id={user_id}, chunk_strategy={chunk_strategy}")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    try:
        service = get_document_service()
        doc = await service.process_upload(file, document_name, user_id, chunk_strategy)
        return {
            "success": True,
            "document": {
                "id": doc.id,
                "document_name": doc.document_name,
                "file_size": doc.file_size,
                "status": doc.status,
                "chunk_strategy": chunk_strategy or "structural,recursive",
            },
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """列出所有已上传的文档"""
    user_id = current_user.get("user_id")
    try:
        service = get_document_service()
        docs = await service.list_documents(user_id)
        return {
            "success": True,
            "documents": [
                {"id": d.id, "user_id": d.user_id, "document_name": d.document_name, "status": d.status}
                for d in docs
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """删除文档"""
    service = get_document_service()
    success = await service.delete_document(document_id)
    return {"success": success}
