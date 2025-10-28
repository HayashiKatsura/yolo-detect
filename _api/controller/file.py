from typing import List, Optional, Any, Dict
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from _api.configuration.NoStandardResponse import ResponseCode, NoStandardResponse  # 按你的实际路径调整
from _api.services.UploadFiles import UploadFiles
from _api.services.DownloadFiles import DownloadFiles
from fastapi.responses import FileResponse




# 依赖：日志装饰器在 FastAPI 里通常写成 dependency
async def log_api_call_dep():
    # TODO: 这里做你的日志记录（等价于 @log_api_call）
    # 例如记录 request.method, request.url, 当前用户等
    return None



router = APIRouter(prefix="/zjut", tags=["Files"])


@router.post("/upload_file/{files_type}")
async def upload_file(
    files_type: str,
    files_data: List[UploadFile] = File(..., alias="files",description="上传的文件列表（字段名 files_data）"),
    folder_id: Optional[str] = Query(None, description="文件ID，可选；若传入则与该文件同目录保存")
) -> JSONResponse:
    """
    上传文件：
    - 若传入 folder_id（= files.id），与该文件同目录保存；
    - 否则在 `_api/data/{prefix}/{timestamp-uuid}` 下新建目录后保存；
    - 同一次请求的多个文件保存到同一目录。
    """
    results = UploadFiles(
        files_type=files_type,
        files_data=files_data,
        folder_id=folder_id
    )._upload_file_local_storage()

    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, "success", data=results).get_response_body()
    )
    
@router.get("/download/{file_id}")
def download_file(file_id: str):
    """
    下载文件
    """
    results = DownloadFiles(file_id)._download_local_storage()
    if results['msg'] !='ok':
        return JSONResponse(
            content=NoStandardResponse(ResponseCode.NOT_FOUND, "下载失败", data=None).get_response_body()
        )
    return FileResponse(
        path=results['path'],
        filename=results['filename'],
        media_type=results['media_type'],
    )