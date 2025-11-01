from enum import Enum
from genericpath import isdir
import os
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(PJ_ROOT)
import time
from uuid import uuid4
from _api._utils.DataRecord import data_record
from _api._utils.UnZip import extract_zip, extract_rar, extract_7z
from abc import ABC, abstractmethod
import tempfile
import csv
import base64
from  _api._utils.ReadCSV import csv_key_to_value
from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType


from  _api._utils.ParseSaveUrl import ParseSaveUrl
from _api._utils.UpdateYaml import update_yaml_config
from _api._utils.ImagestransferComponent.FromLocalImageFiles import TransferLocalImageFiles
from _api._utils.IsFilesExist import FilesID,is_files_exist,files_info
import cv2
import ast

from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error



from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlmodel import select
from _api.models.file import File as FileObject
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from sqlalchemy import func

from _api.configuration.RedisConfig import redis_config
import json
from _api.configuration.MimeTypes import *
from datetime import datetime
import mimetypes
class GetFilesData:
    def __init__(self, file_type: Optional[str] = None, 
                 file_id: Optional[str] = None, 
                 file_name : Optional[str] = None,
                 page: int = 1, page_size: int = 10):
        self.file_type = file_type
        self.file_id = file_id
        self.file_name = file_name
        self.page = page
        self.page_size = page_size
        self.redis_client = redis_config.get_client()
    
    def local_storage(self) -> Dict[str, Any]:
        # 处理单个文件查询
        if self.file_id:
            with Session(SQL_ENGINE) as session:
                file = session.get(FileObject, self.file_id)
                if not file:
                    return {"message": "File not found", "code": 404}
                return {
                    "total": 1,
                    "page": 1,
                    "page_size": 1,
                    "files": [file.to_dict()]
                }
        
        # 构建基础查询
        base_query = select(FileObject).where(FileObject.deleted_at.is_(None))
        count_query = select(func.count()).select_from(FileObject).where(FileObject.deleted_at.is_(None))
        
        # 统一的过滤条件 - 模糊匹配
        if self.file_type:
            # 使用 contains 或 like 进行模糊匹配
            base_query = base_query.where(FileObject.kind.contains(self.file_type))
            count_query = count_query.where(FileObject.kind.contains(self.file_type))
        
        # 计算分页参数
        offset = (self.page - 1) * self.page_size
        
        # 在同一个 session 中执行查询和计数
        with Session(SQL_ENGINE) as session:
            files = session.execute(
                base_query.offset(offset).limit(self.page_size)
            ).scalars().all()
            
            total = session.execute(count_query).scalar()
        
        return {
            "total": total,
            "page": self.page,
            "page_size": self.page_size,
            "files": [file.to_dict() for file in files]
        }
    
    def local_files_show(self):
        storage_path = None
        iterator_func = None
        if self.file_type == 'predicted_image': # 预测图像
            with Session(SQL_ENGINE) as session:
                query = select(FileObject.media_annotations).where(FileObject.id == self.file_id)
                media_annotations = session.execute(query).scalar()
                try:
                    storage_path = media_annotations[0].get("images",None)
                except:
                    pass
        elif self.file_type == 'validated_image': # 验证图像
            
            redis_key = f"validation:{self.file_id}"
            model_metrics = json.loads(self.redis_client.get(redis_key))
            try:
                validated_images= model_metrics[0].get("val_images",None)
                storage_path = next((item.get("image") for item in validated_images if item.get("name") == self.file_name), None)
                if storage_path:
                    return {
                        "path": storage_path,
                        "media_type": get_mime_type(storage_path)
                    }
            except:
                pass
         
            with Session(SQL_ENGINE) as session:
                query = select(FileObject.model_metrics).where(FileObject.id == self.file_id)
                model_metrics = session.execute(query).scalar()
                try:
                    validated_images= model_metrics[0].get("val_images",None)
                    storage_path = next((item.get("image") for item in validated_images if item.get("name") == self.file_name), None)
                except:
                    pass
        
        elif self.file_type =='training_log': # 训练日志
            redis_key = f"train-task:{datetime.now().date()}:{self.file_id}"
            log_name = os.path.basename(self.redis_client.get(redis_key))
            storage_path = os.path.join(PJ_ROOT,'_api/logs/train',f"{log_name}.log")
            def file_iterator():
                with open(storage_path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 64), b""):
                        yield chunk
            iterator_func = file_iterator()
            media_type, _ = mimetypes.guess_type(storage_path)

        else:
            with Session(SQL_ENGINE) as session:
                query = select(FileObject.storage_path,FileObject.original_filename).where(FileObject.id == self.file_id)
                storage_path,file_name = session.execute(query).first()
                media_type = get_mime_type(storage_path)
            
        return {
            "path": storage_path,
            "media_type": media_type or get_mime_type(storage_path),
            "iterator_func": iterator_func
        }
        

    
    def local_validation(self) -> Dict[str, Any]:
        
        # 检查是否有缓存结果
        redis_key = f"validation:{self.file_id}:page:{self.page}:size:{self.page_size}" # TODO Redis端 没有实际分页

        cached_data = self.redis_client.get(redis_key)
        if cached_data:
            # 如果缓存中有数据，直接返回
            print("Found result in Redis cache.")
            return json.loads(cached_data)
        else:
            # 查询单个文件记录，包含 model_metrics 列表
            with Session(SQL_ENGINE) as session:
                file = session.get(FileObject, self.file_id)
                if not file:
                    return {"message": "File not found", "code": 404}
                
                # 获取 model_metrics 数据（这里是一个 JSON 列表）
                model_metrics = file.model_metrics  # 假设 model_metrics 是 JSON 列表
                
                # 计算分页参数
                offset = (self.page - 1) * self.page_size

                # 对 JSON 列表进行分页
                paginated_metrics = model_metrics[offset:offset + self.page_size]
                
                # 返回分页后的数据
                result_data = {
                    "total": len(model_metrics),  # 总记录数
                    "page": self.page,
                    "page_size": self.page_size,
                    "records": [{
                        "file_id": file.id,
                        "file_name": file.original_filename,
                        "model_metrics": paginated_metrics
                    }]
                }

                self.redis_client.setex(redis_key, 3600*24, json.dumps(result_data))

                return result_data
            
    def local_prediction(self) -> Dict[str, Any]:
        
        redis_key = f"prediction:{self.file_id}:page:{self.page}:size:{self.page_size}"

        cached_data = self.redis_client.get(redis_key)
        if cached_data:
            print("Found result in Redis cache.")
            return json.loads(cached_data)
        else:
            with Session(SQL_ENGINE) as session:
                file = session.get(FileObject, self.file_id)
                if not file:
                    return {"message": "File not found", "code": 404}
                
                media_annotations = file.media_annotations  
                
                offset = (self.page - 1) * self.page_size

                paginated_annotations = media_annotations[offset:offset + self.page_size]
                
                result_data = {
                    "total": len(media_annotations), 
                    "page": self.page,
                    "page_size": self.page_size,
                    "records": [{
                        "file_id": file.id,
                        "file_name": file.original_filename,
                        "media_annotations": paginated_annotations
                    }]
                }

                self.redis_client.setex(redis_key, 3600*24, json.dumps(result_data))

                return result_data
        
    def cloud_storage(self):
        pass
    
    def cloud_validation(self):
        pass
    
    def cloud_prediction(self):
        pass

