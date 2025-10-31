import os
# 获取当前项目所在路径
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(PJ_ROOT)

import zipfile
import tempfile
import shutil
import io
from _api._utils.IsFilesExist import FilesID,is_files_exist,files_info
from uuid import uuid4
import time
from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType
import ast

from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error
from fastapi import APIRouter, HTTPException, Path, Query
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from sqlalchemy.orm import Session
from _api.models.file import File as FileObject
from _api.configuration.MimeTypes import *

# 文件类型映射
from io import BytesIO
from typing import List, Union
from sqlmodel import Session
from datetime import datetime




def safe_join(base: str, *paths: str) -> str:
    """防止路径逃逸"""
    final_path = os.path.abspath(os.path.join(base, *paths))
    base_abs = os.path.abspath(base)
    if not final_path.startswith(base_abs):
        raise HTTPException(status_code=400, detail="非法文件路径")
    return final_path

# class DownloadFiles:
    
#     def __init__(self,file_ids):
#         self.file_ids = file_ids
    
    
#     def _download_local_storage(self):
#         """
#         下载本地文件
#         """
#         files_info = []
#         with Session(SQL_ENGINE) as session:
#             for file_id in self.file_ids:
#                 file = session.get(FileObject, file_id)
#                 if file:
#                     files_info.append({
#                         "id": file.id,
#                         "filename": file.original_filename,
#                         "media_type":get_mime_type(file.original_filename),
#                         "size": file.size_bytes,
#                         "path": file.storage_path,
#                     })
#         return files_info
    
#     def _download_cloud_storage(self):
#         """
#         下载云端文件
#         """
#         pass

    
    


class DownloadFiles:
    def __init__(self, file_ids: List[Union[str, int]]):
        self.file_ids = file_ids
    
    def _get_files_info(self):
        """
        获取文件信息列表（用于前端分别下载）
        """
        files_info = []
        
        with Session(SQL_ENGINE) as session:
            for file_id in self.file_ids:
                file = session.get(FileObject, file_id)
                
                if not file:
                    continue
                
                abs_path = safe_join(PJ_ROOT, file.storage_path)
                if not os.path.exists(abs_path):
                    continue
                
                try:
                    file_size = os.path.getsize(abs_path)
                except:
                    file_size = 0
                
                files_info.append({
                    "id": file.id,
                    "filename": file.original_filename or os.path.basename(abs_path),
                    "size": file_size,
                    "url": f"/download-file/{file.id}"
                })
        
        return files_info
    
    def _download_single_file(self):
        """
        下载单个文件
        """
        if not self.file_ids:
            return {'msg': "文件ID不能为空"}
        
        file_id = self.file_ids[0]
        
        with Session(SQL_ENGINE) as session:
            file = session.get(FileObject, file_id)
        
        if not file:
            return {'msg': "文件不存在"}

        abs_path = safe_join(PJ_ROOT, file.storage_path)
        if not os.path.exists(abs_path):
            return {'msg': "文件路径不存在"}
        
        file_name = file.original_filename or os.path.basename(abs_path)
        mime_type = get_mime_type(file_name)
        
        return {
            'msg': "ok",
            'path': abs_path,
            'filename': file_name,
            'media_type': mime_type,
        }
    
    def _download_as_zip(self):
        """
        批量下载，打包成ZIP
        
        针对图片等已压缩文件使用 ZIP_STORED（不压缩），速度更快
        """
        if not self.file_ids:
            return {'msg': "文件ID不能为空"}
        
        # 创建内存中的ZIP文件
        zip_buffer = BytesIO()
        
        # 判断文件类型，选择压缩方式
        compression = zipfile.ZIP_STORED  # 默认不压缩（适合图片、视频等）
        
        with zipfile.ZipFile(zip_buffer, 'w', compression) as zip_file:
            with Session(SQL_ENGINE) as session:
                for file_id in self.file_ids:
                    file = session.get(FileObject, file_id)
                    
                    if not file:
                        continue  # 跳过不存在的文件
                    
                    abs_path = safe_join(PJ_ROOT, file.storage_path)
                    if not os.path.exists(abs_path):
                        continue
                    
                    # 使用原始文件名添加到ZIP
                    file_name = file.original_filename or os.path.basename(abs_path)
                    
                    # 处理重名文件
                    arcname = file_name
                    counter = 1
                    while arcname in zip_file.namelist():
                        name, ext = os.path.splitext(file_name)
                        arcname = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    try:
                        zip_file.write(abs_path, arcname=arcname)
                    except Exception as e:
                        print(f"添加文件失败 {file_name}: {str(e)}")
                        continue
        
        # 检查ZIP是否为空
        zip_buffer.seek(0)
        zip_size = len(zip_buffer.getvalue())
        
        if zip_size == 0:
            return {'msg': "没有找到有效的文件"}
        
        zip_buffer.seek(0)
        
        # 生成ZIP文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"files_{timestamp}.zip"
        
        return {
            'msg': "ok",
            'zip_buffer': zip_buffer,
            'filename': zip_filename,
        }
