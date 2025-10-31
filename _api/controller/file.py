from typing import List, Optional, Any, Dict , Union
from fastapi import APIRouter, UploadFile, File, Query, HTTPException , Body
from fastapi.responses import JSONResponse,FileResponse, StreamingResponse
from _api.configuration.NoStandardResponse import ResponseCode, NoStandardResponse  # 按你的实际路径调整
from _api.services.UploadFiles import UploadFiles
from _api.services.DeleteFiles import DeleteFiles

from _api.services.DownloadFiles import DownloadFiles
from fastapi.responses import FileResponse
from _api.services.GetFilesData import GetFilesData
from _api.configuration.RequestBody import *



# 依赖：日志装饰器在 FastAPI 里通常写成 dependency
async def log_api_call_dep():
    # TODO: 这里做你的日志记录（等价于 @log_api_call）
    # 例如记录 request.method, request.url, 当前用户等
    return None



router = APIRouter(prefix="/zjut", tags=["Files"])


@router.post("/upload-files/{files_type}")
async def upload_files(
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

@router.delete("/delete-files")
async def delete_files(
    request:DeleteFilesRequest,
) -> JSONResponse:
    """
    删除文件
    - 删除多个文件
    """
    results = DeleteFiles(
        file_ids=request.file_ids
    )._delete_files_local_storage()

    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, "success", data=results).get_response_body()
    )

# @router.get("/download-files/{file_id}")
# def download_file(
#     request:DeleteFilesRequest,
# ):
#     """
#     下载文件
#     """
#     results = DownloadFiles(
#         file_ids=request.file_ids
#     )._download_local_storage()
#     if results['msg'] !='ok':
#         return JSONResponse(
#             content=NoStandardResponse(ResponseCode.NOT_FOUND, "下载失败", data=None).get_response_body()
#         )
#     return FileResponse(
#         path=results['path'],
#         filename=results['filename'],
#         media_type=results['media_type'],
#     )


@router.get("/download-file/{file_id}")
def download_single_file(file_id: Union[str, int]):
    """
    下载单个文件
    """
    try:
        downloader = DownloadFiles(file_ids=[file_id])
        result = downloader._download_single_file()
        
        if result['msg'] != 'ok':
            raise HTTPException(status_code=404, detail=result['msg'])
        
        return FileResponse(
            path=result['path'],
            filename=result['filename'],
            media_type=result['media_type'],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download-files")
def download_multiple_files(
    file_ids: List[Union[str, int]] = Body(..., embed=True, description="要下载的文件ID列表"),
    force_zip: bool = Body(False, description="强制打包成ZIP")
):
    """
    智能批量下载文件
    
    策略：
    - 1个文件：直接下载
    - 2-3个文件且不强制：返回文件信息，前端分别下载
    - 4个以上文件或强制：打包成ZIP下载
    """
    if not file_ids:
        raise HTTPException(status_code=400, detail="文件ID列表不能为空")
    
    try:
        # 单文件直接下载
        if len(file_ids) == 1 and not force_zip:
            return download_single_file(file_ids[0])
        
        downloader = DownloadFiles(file_ids=file_ids)
        
        # 少量文件且不强制打包，返回文件信息让前端并发下载
        if len(file_ids) <= 3 and not force_zip:
            file_info = downloader._get_files_info()
            if not file_info:
                raise HTTPException(status_code=404, detail="没有找到有效的文件")
            
            return JSONResponse(
                content={
                    "type": "separate",
                    "files": file_info,
                    "message": "建议分别下载"
                }
            )
        
        # 多文件或强制打包，返回ZIP
        result = downloader._download_as_zip()
        
        if result['msg'] != 'ok':
            raise HTTPException(status_code=404, detail=result['msg'])
        
        return StreamingResponse(
            result['zip_buffer'],
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={result['filename']}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@router.post("/download-files-zip")
def download_files_as_zip(
    file_ids: List[Union[str, int]] = Body(..., embed=True, description="要下载的文件ID列表")
):
    """
    强制打包成ZIP下载（适合大量文件）
    """
    if not file_ids:
        raise HTTPException(status_code=400, detail="文件ID列表不能为空")
    
    try:
        downloader = DownloadFiles(file_ids=file_ids)
        result = downloader._download_as_zip()
        
        if result['msg'] != 'ok':
            raise HTTPException(status_code=404, detail=result['msg'])
        
        return StreamingResponse(
            result['zip_buffer'],
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={result['filename']}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"打包失败: {str(e)}")








import mimetypes
import os
@router.get("/show-files/{file_id}")
async def show_files(
    file_id: str,
    file_type: Optional[str] = Query(None, description="文件类型，未传则获取所有类型文件，否则获取指定类型文件"),
    file_name: Optional[str] = Query(None, description="文件名，未传则获取所有文件，否则获取指定文件"),
     t: Optional[str] = Query(None, description="缓存控制时间戳（忽略）")
) -> JSONResponse:
    """
    获取指定文件类型的文件数据，支持分页。
    - `file_type`：文件类型，未传则获取所有类型文件，否则获取指定类型文件
    - `file_id`：文件id，若传入，返回指定文件的数据
    - `page`：页码，默认值为 1
    - `page_size`：每页数量，默认 10，最大值为 100
    """
    # file_path = '/Users/katsura/Documents/code/ultralytics-12/_api/logs/train/202510311858-Test.log'
    # mime_type, _ = mimetypes.guess_type(file_path)
    # if (mime_type and mime_type.startswith("text")) or file_path.endswith((".log", ".txt", ".yaml", ".yml", ".json")):
    # # 二进制读取，避免换行被破坏
    #     def file_iterator():
    #         with open(file_path, "rb") as f:
    #             for chunk in iter(lambda: f.read(1024 * 64), b""):
    #                 yield chunk

    # return StreamingResponse(
    #     file_iterator(),
    #     media_type=f"{mime_type or 'text/plain'}; charset=utf-8",
    #     headers={
    #         "Content-Disposition": f'inline; filename="{os.path.basename(file_path)}"'
    #     },
    # )
    
    
    
    
    
    results = GetFilesData(
        file_type=file_type,
        file_id=file_id,
        file_name=file_name
    ).local_files_show()
    
    iterator_func = results.get('iterator_func',None)
    if iterator_func:
        return StreamingResponse(
        iterator_func,
        media_type=f"{results['media_type'] or 'text/plain'}; charset=utf-8",
        headers={
            "Content-Disposition": f'inline; filename="{os.path.basename(results.get("path"))}"'
        },
    )

    return FileResponse(
        path=results['path'],
        media_type=results['media_type'],
    )