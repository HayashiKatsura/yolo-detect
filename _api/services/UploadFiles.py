from enum import Enum
from genericpath import isdir, isfile
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
import shutil

from  _api._utils.ParseSaveUrl import ParseSaveUrl
from _api._utils.ArchiveExtractor import ArchiveExtractor,extract_archive
from _api._utils.RWYaml import update_yaml_keys,read_yaml
from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error
from _api._utils.RWYaml import read_yaml

from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType,table_1, table_2,table_3,table_4

from _project.mydata._code._utils.files_process.txt_process import remove_last_column_in_txt
from sqlmodel import select,update


import re
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, UploadFile, File as FAFile, Query, HTTPException
from fastapi.responses import JSONResponse

from sqlmodel import Session
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from _api.models.file import File as FileObject
from _api._utils.ArchiveExtractor import ArchiveExtractor,extract_archive
import glob
from _api._utils.RWYaml import read_yaml
from threading import Thread
from datetime import datetime

def _secure_filename(name: str) -> str:
    """
    简单文件名清洗：去掉路径成分与危险字符，避免目录穿越/奇怪符号
    """
    name = os.path.basename(name)
    # 只保留常见字符，其余替换为下划线
    name = re.sub(r'[^A-Za-z0-9.\-\u4e00-\u9fa5_]+', '_', name)
    # 避免全空
    return name or f"file_{uuid4().hex}"


def _unique_path(dirpath: str, filename: str) -> str:
    """
    如果文件已存在，则自动追加序号避免覆盖
    e.g. a.jpg -> a(1).jpg -> a(2).jpg ...
    """
    base, ext = os.path.splitext(filename)
    candidate = filename
    idx = 1
    full = os.path.join(dirpath, candidate)
    while os.path.exists(full):
        candidate = f"{base}({idx}){ext}"
        full = os.path.join(dirpath, candidate)
        idx += 1
    return full

def format_size(size: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

# TODO 需要一些更严校验判断
def datasets_validate(zip_save_path: str):
    def content_struct_validate(dataset_folder: str):
        yaml_config_list = glob.glob(os.path.join(dataset_folder,'*.yaml')) or glob.glob(os.path.join(dataset_folder,'*.yml'))
        if len(yaml_config_list)!=1:
            return {}
        
        for images_labels in ['images','labels']:
            for train_val in ['train','val']:
                if not os.path.exists(os.path.join(dataset_folder,images_labels,train_val)):
                    return {}
        yaml_path = yaml_config_list[0]
        new_names = read_yaml(yaml_path).get('names',None)
        new_names = None if new_names=={} else new_names
        
        train_folder = os.path.join(dataset_folder,'images','train')
        val_folder = os.path.join(dataset_folder,'images','val')
        
        train_count = len([f for f in glob.glob(os.path.join(train_folder, "*.*")) if os.path.isfile(f) and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']])
        val_count = len([f for f in glob.glob(os.path.join(val_folder, "*.*")) if os.path.isfile(f) and os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']])
        
        ctx_list=[
                {'path':dataset_folder},
                {'train':train_folder},
                {'val':val_folder},
                {'names':new_names}
            ]
        update_yaml_keys(
            file_path=str(yaml_path),
            ctx_list=ctx_list
        )
        results = {
            'name':os.path.basename(dataset_folder),
            'yaml_path':yaml_path,
            'train_cases_count': train_count,
            'val_cases_count': val_count,
            'last_train_cases_count':train_count,
            'last_val_cases_count':val_count,
            'train_count':0,
            'train_date':None,
            'metrics':None
            
        }
        for item in ctx_list:
            results.update(item)
        return results
        
    
    extract_archive(zip_save_path,os.path.dirname(zip_save_path))
    os.remove(zip_save_path)
    unzip_dir,probable_dataset_dir,real_dataset_dir = os.path.dirname(zip_save_path),None,None
    
    folder_count = 0
    if os.path.exists(MAC_TRASH:=os.path.join(unzip_dir,'__MACOSX')):
        shutil.rmtree(MAC_TRASH)
    for _unzip_dataset in os.listdir(unzip_dir):
        if os.path.isdir(os.path.join(unzip_dir,_unzip_dataset)):
            probable_dataset_dir = os.path.join(unzip_dir,_unzip_dataset)
            break 
    
    folder_count = len([name for name in os.listdir(probable_dataset_dir) if os.path.isdir(os.path.join(probable_dataset_dir, name))])
    if folder_count == 1: # 可能是嵌套文件夹，最多只允许套一层，超过一层，不再校验
        real_dataset_dir = [name for name in os.listdir(probable_dataset_dir) if os.path.isdir(os.path.join(probable_dataset_dir, name))][0]
    else: 
        real_dataset_dir = probable_dataset_dir
        
    return content_struct_validate(real_dataset_dir)        

        
class UploadFiles:
    """
    上传文件到本地存储
    - 如果给了 folder_id（= files.id），则复用该文件所在目录
    - 否则按 file_type + 时间戳创建新目录
    - 多文件一次请求全部写入同一目录
    """

    def __init__(self, files_type: str, files_data: List[UploadFile]):
        self.time_stamp = time.strftime("%Y%m%d%H%M", time.localtime())
        self.files_type = files_type
        self.files_data = files_data
        self.temp_folder = os.path.join(PJ_ROOT, '_api/data/temp',f"temp-{self.files_type}-{self.time_stamp}")


    def _upload_file_local_storage(self,folder_id=None) -> Dict[str, Any]:
        saved: List[Dict[str, Any]] = []

        if folder_id:
            if self.files_type == 'datasets': # 更新数据集
                with Session(SQL_ENGINE) as session:
                    query = select(FileObject.remark).where(FileObject.id==folder_id and FileObject.deleted_at.is_(None))
                    datasets_info = session.execute(query).scalar()
                
                os.makedirs(self.temp_folder, exist_ok=True)
                
                for uf in self.files_data: # 前端已做限制，只允许传一个
                    original_name = _secure_filename(uf.filename or "")
                    full_path = _unique_path(self.temp_folder, original_name)

                    with open(full_path, "wb") as out:
                        shutil.copyfileobj(uf.file, out, length=100 * 1024 * 1024)
                    
                    supplement = datasets_validate(full_path)
                    
                    original_path,supplement_path = datasets_info['path'],supplement['path']
                    original_train_cases_count,original_val_cases_count = datasets_info['train_cases_count'],datasets_info['val_cases_count']
                    supplement_train_cases_count,supplement_val_cases_count = supplement['train_cases_count'],supplement['val_cases_count']
                    
                    saved.append({
                        "file_id": folder_id,
                        "original_filename": original_name,
                        "size_bytes": format_size(os.path.getsize(os.path.dirname(full_path))),              
                    })
                    
                    def copy_files():
                        for train_val in ['train','val']:
                            try:
                                supplement_images_folder = os.path.join(supplement_path,'images',train_val)
                                supplement_labels_folder = os.path.join(supplement_path,'labels',train_val)
                                for file in os.listdir(supplement_images_folder):
                                    if os.path.exists(os.path.join(original_path,'images',train_val, file)):
                                        shutil.copyfile(os.path.join(supplement_images_folder,file),os.path.join(original_path,'images',train_val,f"{self.time_stamp}-{file}")) # 图像
                                        shutil.copyfile(os.path.join(supplement_labels_folder,f"{os.path.splitext(file)[0]}.txt"),os.path.join(original_path,'labels',train_val,f"{self.time_stamp}-{os.path.splitext(file)[0]}.txt")) # 标签
                                    else:
                                        shutil.copyfile(os.path.join(supplement_images_folder,file),os.path.join(original_path,'images',train_val,file))
                                        shutil.copyfile(os.path.join(supplement_labels_folder,f"{os.path.splitext(file)[0]}.txt"),os.path.join(original_path,'labels',train_val,f"{os.path.splitext(file)[0]}.txt"))
                            except:
                                continue
                        shutil.rmtree(supplement_path) # TODO 不排除存在并发冲突的问题
                                    
                    def update_db():
                        # 修改原数据集的信息
                        datasets_info['last_train_cases_count'] = datasets_info['train_cases_count']
                        datasets_info['last_val_cases_count'] = datasets_info['val_cases_count']
                        datasets_info['train_cases_count'] = original_train_cases_count + supplement_train_cases_count
                        datasets_info['val_cases_count'] = original_val_cases_count + supplement_val_cases_count
                        
                        with Session(SQL_ENGINE) as session:
                            try:
                                query = update(FileObject).where(FileObject.id==folder_id).values(
                                    remark=datasets_info,
                                    updated_at=datetime.utcnow()
                                )
                                session.execute(query)
                                session.commit()
                            except Exception as e:
                                print(e)
                                session.rollback()
                                     
                    t1 = Thread(target=copy_files)
                    t2 = Thread(target=update_db)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                    
                    return {
                        "count": len(saved),
                        "files": saved
                    }

            else: # 其他定制需求
                pass
                    
        else:
            # 未指定 folder_id：自动创建新目录
            folder_prefix = self.files_type.split('_')[0] if '_' in str(self.files_type) else str(self.files_type)
            rel_dir = os.path.join("_api", "data", folder_prefix, f"{self.time_stamp}-{uuid4()}")
            self.save_folder = os.path.join(PJ_ROOT, rel_dir)  # 实际磁盘目录
            os.makedirs(self.save_folder, exist_ok=True)
        
        # 普通上传
        with Session(SQL_ENGINE) as session:
            try:
                for uf in self.files_data:
                    original_name = _secure_filename(uf.filename or "")
                    full_path = _unique_path(self.save_folder, original_name)

                    with open(full_path, "wb") as out:
                        shutil.copyfileobj(uf.file, out, length=100 * 1024 * 1024)
                    
                    size_bytes = format_size(os.path.getsize(full_path))  
                    
                    remark = None 
                    if str(self.files_type).find('dataset') !=-1:
                        # 如果是数据集文件，需要特殊处理
                        datasets_info = datasets_validate(full_path)
                        original_name = datasets_info.get('name')
                        full_path = datasets_info.get('path')
                        if not datasets_info:
                            raise HTTPException(status_code=400, detail="上传文件格式错误")
                        remark = datasets_info
                        
                    new_file = FileObject(
                        kind=self.files_type,                      
                        content_type=uf.content_type or None,      # MIME
                        original_filename=original_name,
                        storage_path=full_path,                    
                        size_bytes=size_bytes,                 
                        remark=remark,
                    )
                    session.add(new_file)
                    session.flush()  # ✅ 获取自增主键 id（不提交事务）

                    saved.append({
                        "file_id": new_file.id,
                        "original_filename": original_name,
                        "size_bytes": size_bytes,              
                    })

                session.commit()  # ✅ 统一提交
            except:
                session.rollback()

        return {
            "count": len(saved),
            "files": saved
        }
        
 
    
    
    
    
    
    