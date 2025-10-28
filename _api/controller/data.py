from typing import List, Optional, Any, Dict
from fastapi import APIRouter, UploadFile, File as FAFile, Query, HTTPException
from fastapi.responses import JSONResponse
from _api.configuration.NoStandardResponse import ResponseCode, NoStandardResponse  # 按你的实际路径调整
from _api.services.GetFiles import GetFiles




router = APIRouter(prefix="/zjut", tags=["Data"])

@router.get("/data")
async def get_files(
    file_type: Optional[str] = Query(None, description="文件类型，未传则获取所有类型文件，否则获取指定类型文件"),
    file_id: Optional[str] = Query(None, description="文件id"),
    page: Optional[int] = Query(1, ge=1, description="页码（最小值为 1）"),
    page_size: Optional[int] = Query(10, ge=1, le=100, description="每页数量，最大为 100")
) -> JSONResponse:
    """
    获取指定文件类型的文件数据，支持分页。
    - `file_type`：文件类型，未传则获取所有类型文件，否则获取指定类型文件
    - `file_id`：文件id，若传入，返回指定文件的数据
    - `page`：页码，默认值为 1
    - `page_size`：每页数量，默认 10，最大值为 100
    """
    results = GetFiles(
        file_type=file_type,
        file_id=file_id,
        page=page,
        page_size=page_size
    ).local_storage()

    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, "success", data=results).get_response_body()
    )