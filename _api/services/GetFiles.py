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

class GetFiles:
    def __init__(self, file_type: Optional[str] = None, file_id: Optional[str] = None, 
                 page: int = 1, page_size: int = 10):
        self.file_type = file_type
        self.file_id = file_id
        self.page = page
        self.page_size = page_size
    
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
        base_query = select(FileObject)
        count_query = select(func.count()).select_from(FileObject)
        
        # 统一的过滤条件 - 模糊匹配
        if self.file_type:
            # 使用 contains 或 like 进行模糊匹配
            base_query = base_query.where(FileObject.kind.contains(self.file_type) and FileObject.deleted_at.is_(None))
            count_query = count_query.where(FileObject.kind.contains(self.file_type) and FileObject.deleted_at.is_(None))
            
            # 或者使用 like (需要手动添加 %)
            # base_query = base_query.where(FileObject.kind.like(f"%{self.file_type}%"))
            # count_query = count_query.where(FileObject.kind.like(f"%{self.file_type}%"))
        
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
    def cloud_oss(self):
        pass



# class GetFileType(Enum):
#     """
#     获取文件类型
#     """
#     image = 'image'
#     weight = 'weight'
#     compressed = 'compressed'
#     dataset = 'dataset'

#     def __str__(self):
#         return self.value


# class GetFiles(ABC):
#     def __init__(self, file_id):
#         self.file_id = file_id # 文件id
#     @abstractmethod
#     def _images(self):
#         pass
#     @abstractmethod
#     def _weights(self):
#         pass
#     # @abstractmethod 
#     # def _compressed(self):
#     #     pass
#     @abstractmethod
#     def _datasets(self):
#         pass
    

# class GetFilesLocalStorage(GetFiles):
#     """
#     获取文件到本地存储
#     """
#     def __init__(self, file_id=None):
#         super().__init__(file_id)
#         self.file_id = file_id # 二进制文件


#     def _images(self):
#         """
#         获取图像文件
#         """
#         file_type = '' # 文件类型
#         is_exist = False # 是否存在
#         file_path = '' # 实际路径
#         real_name = '' # 文件名
#         file_path = csv_key_to_value(
#             source = {
#                 'key':'file_id', 
#                 'value': str(self.file_id),
#                 'csv': str(os.path.join(PJ_ROOT, '_api/data/data_record.csv')),
#                 },
#             target='file_path')
#         if file_path:
#             if os.path.exists(file_path):
#                 is_exist = True
#                 file_type = 'file' if os.path.isfile(file_path) else 'folder'
#                 real_name = os.path.basename(file_path)
#             else:
#                 is_exist = False
#         else:
#             is_exist = False
        
#         return {
#             'file_id': self.file_id,
#             'file_path': file_path,
#             'file_real_name': real_name,
#             'is_exist': is_exist,
#             'file_type': file_type,
#         }
    
#     def _weights(self):
#         """
#         获取模型权重
#         """
#         file_type = '' # 文件类型
#         is_exist = False # 是否存在
#         file_path = '' # 实际路径
#         real_name = '' # 文件名
#         file_path = csv_key_to_value(
#             source = {
#                 'key':'file_id', 
#                 'value': str(self.file_id),
#                 'csv': str(os.path.join(PJ_ROOT, '_api/data/data_record.csv')),
#                 },
#             target='file_path')
#         if file_path:
#             if os.path.exists(file_path):
#                 is_exist = True
#                 file_type = 'file' if os.path.isfile(file_path) else 'folder'
#                 real_name = os.path.basename(file_path)
#             else:
#                 is_exist = False
#         else:
#             is_exist = False
        
#         return {
#             'file_id': self.file_id,
#             'file_path': file_path,
#             'file_real_name': real_name,
#             'is_exist': is_exist,
#             'file_type': file_type,
#         }
    
    
#     def _datasets(self):
#         """
#         获取数据集
#         """
#         file_type = '' # 文件类型
#         is_exist = False # 是否存在
#         file_path = '' # 实际路径
#         real_name = '' # 文件名
#         file_path = csv_key_to_value(
#             source = {
#                 'key':'file_id', 
#                 'value': str(self.file_id),
#                 'csv': str(os.path.join(PJ_ROOT, '_api/data/data_record.csv')),
#                 },
#             target='file_path')
#         if file_path:
#             if os.path.exists(file_path):
#                 is_exist = True
#                 file_type = 'file' if os.path.isfile(file_path) else 'folder'
#                 real_name = os.path.basename(file_path)
#             else:
#                 is_exist = False
#         else:
#             is_exist = False
        
#         return {
#             'file_id': self.file_id,
#             'file_path': file_path,
#             'file_real_name': real_name,
#             'is_exist': is_exist,
#             'file_type': file_type,
#         }
        
        
#     def _check_files_local_table(self):
#         """
#         获取当前的文件信息
#         """
#         LOCAL_DATA_TABLE = os.path.join(PJ_ROOT, '_api/data/data_record.csv') 
#             # 存储所有存在的文件
#         valid_files = []

#         # 打开CSV文件并读取数据
#         with open(LOCAL_DATA_TABLE, mode='r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
            
#             # 遍历每一行
#             for row in reader:
#                 # 获取file_path
#                 file_path = row.get('file_path')
                
#                 # 检查文件是否存在
#                 if os.path.exists(file_path):
#                     # 文件存在，将当前行的数据（字典）加入valid_files列表
#                     valid_files.append(row)
        
#         return valid_files
    
    
#     @staticmethod
#     @handle_db_error
#     def show_images(file_id,range=None,tag=None):
#         import re
#         if str(file_id).find('image') != -1:
#             try:
#                 # images_info = files_info("file_id",file_id,record_path=files_record_mapping()[str(FilesType.images)][1])
#                 # images_path = images_info.get('file_path')
#                 images_info = FileTable.query.filter(FileTable.file_id == file_id).first()    
#             except:
#                 # images_info = files_info("file_id",file_id,record_path=files_record_mapping()[str(FilesType.cameras)][1])
#                 # images_path = images_info.get('file_path')
#                 images_info = FileTable.query.filter(FileTable.file_id == file_id).first()  
#             images_path = images_info.file_path
                
#             image_url = f'data:image/png;base64,{TransferLocalImageFiles(images_path).toBase64()}'
            
#             try:
#                 # detect_image_id, detect_txt_id = images_info.get('is_detected',None).split(',')
#                 # detect_infomation = ast.literal_eval(
#                 #     files_info("file_id",
#                 #                file_id,
#                 #                record_path=files_record_mapping()[str(FilesType.detect)][1]).get('is_detected'))
                
#                 # detect_image_path,detect_txt = detect_infomation[detect_image_id],detect_infomation[detect_txt_id]
#                 results = []
#                 detect_infomation = DetectionTable.query.filter(DetectionTable.file_id == file_id).first().details
#                 detect_image_path = detect_infomation.get('detect_image_path',None)
#                 detect_txt = detect_infomation.get('detect_txt_path',None)
                
#                 detect_url = f'data:image/png;base64,{TransferLocalImageFiles(detect_image_path).toBase64()}' if detect_image_path else None
                
#                 # preix,_ = os.path.splitext(detect_image_path)
#                 # detect_txt = f'{preix}.txt'
#                 height, width = cv2.imread(images_path).shape[:2]
#                 with open(detect_txt, 'r') as f:
#                     lines = f.readlines()
#                     for line in lines:
#                         _cls,_x,_y,_w,_h,_conf = line.strip().split(' ')
#                         # YOLO 转为左上角坐标
#                         x1 = int((float(_x) - float(_w) / 2) * width)
#                         y1 = int((float(_y) - float(_h) / 2) * height)
#                         x2 = int((float(_x) + float(_w) / 2) * width)
#                         y2 = int((float(_y) + float(_h) / 2) * height)
#                         # 计算面积
#                         detect_area = abs(x1-x2)*abs(y1-y2)
#                         results.append(
#                             {
#                                 'file_name':str(os.path.basename(detect_image_path)),
#                                 'cls':_cls,
#                                 'conf':_conf,
#                                 'yolo_coord':f'(x:{round(float(_x),2)},y:{round(float(_y),2)},w:{round(float(_w),2)},h:{round(float(_h),2)})',
#                                 'detect_coord':f'(x1:{x1},y1:{y1},x2:{x2},y2:{y2})',
#                                 'detect_area':detect_area,
#                                 'image_size':f'height:{height},width:{width}',
#                                 'detect_image_base64':detect_url,
#                                 'file_path':str(os.path.dirname(detect_image_path))
#                             }
#                         )
#             except Exception as e:
#                 detect_url,results = None,None
                
#             return {"image_url":image_url,"detect_url":detect_url,"detect_result":results}
        
#         if str(file_id).find('video') != -1:
#             if tag == "video":
#                 videos_info = FileTable.query.filter(FileTable.file_id == file_id).first()  
#                 videos_path = videos_info.file_path
#             elif tag == "detectVideo":
#                 videos_info = DetectionTable.query.filter(DetectionTable.file_id == file_id).first()
#                 videos_path = videos_info.details.get('detect_image_path',None)
#             if not videos_path:
#                 videos_info = FileTable.query.filter(DetectionTable.file_id == file_id).first()  
#                 videos_path = videos_info.details['detect_image_path']
            
#             file_size = os.path.getsize(videos_path)
#             if not range:
#                 return {"videos_path":videos_path}
            
#             # 解析 Range 请求
#             byte_start, byte_end = 0, file_size - 1
#             match = re.search(r'bytes=(\d+)-(\d*)', range)
            
#             if match:
#                 byte_start = int(match.group(1))
#                 if match.group(2):
#                     byte_end = int(match.group(2))
            
#             # 读取指定范围的数据
#             length = byte_end - byte_start + 1
            
#             with open(videos_path, 'rb') as f:
#                 f.seek(byte_start)
#                 data = f.read(length)
            
#             return {"videos_data":data,'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}','Accept-Ranges': 'bytes','Content-Length': str(length)}
            

#         if str(file_id).find('weight') != -1:
#             # val_floder = files_info("file_id", 
#             #                         file_id,
#             #                         record_path=files_record_mapping()[str(FilesType.weights)][1]).get('is_detected')
            
#             val_floder = WeightTable.query.filter(WeightTable.file_id == file_id).first().is_validated
#             # metrics
#             metrics_reults = {}
#             val_images = []
#             if not os.path.exists(val_floder) or not os.path.isdir(val_floder):
#                 return {'metrics':metrics_reults,'val_images':val_images}
            
#             with open(os.path.join(val_floder,'metrics.txt'), 'r') as f:
#                 metrics = f.readlines()
#                 for metric in metrics:
#                     metric_name, metric_value = metric.strip().split(': ')
#                     metrics_reults.update({metric_name:metric_value})
            
#             # val_image
#             file_name_list = [
#                         'BoxF1_curve.png',
#                         'BoxP_curve.png',
#                         'BoxR_curve.png',
#                         'BoxPR_curve.png',
#                         'confusion_matrix_normalized.png',
#                         'confusion_matrix.png',
#                         'F1_curve.png',
#                         'P_curve.png',
#                         'R_curve.png',
#                         'PR_curve.png',
#                         'val_batch0_labels.jpg',
#                         'val_batch0_pred.jpg',
#                         'val_batch1_labels.jpg',
#                         'val_batch1_pred.jpg',
#                         'val_batch2_labels.jpg',
#                         'val_batch2_pred.jpg',
#                          ]

#             val_images = [
#                 f"data:image/png;base64,{TransferLocalImageFiles(os.path.join(val_floder,'val',file_name)).toBase64()}" 
#                 for file_name in file_name_list if os.path.exists(_image:=os.path.join(val_floder,'val',file_name)) and os.path.isfile(_image) 
#             ]
            
#             return {'metrics':metrics_reults,'val_images':val_images}
            
#         if str(file_id).find('train') != -1:
#             # train_floder = files_info("file_id", 
#             #                         file_id,
#             #                         record_path=files_record_mapping()[str(FilesType.weights)][1]).get('file_path')
#             train_floder = WeightTable.query.filter(WeightTable.file_id == file_id).first().file_path
#             # train_image
#             file_name_list = [
#                         'BoxF1_curve.png',
#                         'BoxP_curve.png',
#                         'BoxPR_curve.png',
#                         'BoxR_curve.png',
#                         'confusion_matrix_normalized.png',
#                         'confusion_matrix.png',
#                         'labels_correlogram.jpg',
#                         'labels.jpg',
#                         'results.png',
#                         'train_batch0.jpg',
#                         'train_batch1.jpg',
#                         'train_batch2.jpg',
#                         'val_batch0_labels.jpg',
#                         'val_batch0_pred.jpg',
#                          ]
#             for sub_folder in os.listdir(train_floder):
#                 if os.path.isdir(save_dir:=os.path.join(train_floder,sub_folder)):
#                     train_images = [
#                         f"data:image/png;base64,{TransferLocalImageFiles(os.path.join(save_dir,file_name)).toBase64()}"
#                         for file_name in file_name_list if os.path.exists(_image:=os.path.join(save_dir,file_name)) and os.path.isfile(_image) 
#                     ]
#                     return {'train_images':train_images}
            
#             return {'train_images':[]}
                    
        
     
                    
       

# class GetFilesCloudOSS(GetFiles):
#     """
#     获取文件到云存储
#     """
#     pass

# if __name__ == '__main__':
#     # def _check_files_local_table():
#     #     """
#     #     获取当前的文件信息
#     #     """
#     #     LOCAL_DATA_TABLE = os.path.join(PJ_ROOT, '_api/data/data_record.csv') 
#     #         # 存储所有存在的文件
#     #     valid_files = []

#     #     # 打开CSV文件并读取数据
#     #     with open(LOCAL_DATA_TABLE, mode='r', encoding='utf-8') as file:
#     #         reader = csv.DictReader(file)
            
#     #         # 遍历每一行
#     #         for row in reader:
#     #             # 获取file_path
#     #             file_path = row.get('file_path')
                
#     #             # 检查文件是否存在
#     #             if os.path.exists(file_path):
#     #                 # 文件存在，将当前行的数据（字典）加入valid_files列表
#     #                 valid_files.append(row)
        
#     #     return valid_files
    
#     # _check_files_local_table()
#     from collections import defaultdict
    
#     def _check_files_local_table():
#         """
#         获取当前的文件信息并按file_folder_id分组
#         """
#         PJ_ROOT = "."  # 根据你的项目调整这个路径
#         LOCAL_DATA_TABLE = os.path.join(PJ_ROOT, '_api/data/data_record.csv') 
        
#         # 存储所有存在的文件
#         valid_files = []

#         # 打开CSV文件并读取数据
#         with open(LOCAL_DATA_TABLE, mode='r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
            
#             # 遍历每一行
#             for row in reader:
#                 # 获取file_path
#                 file_path = row.get('file_path')
                
#                 # 检查文件是否存在
#                 if os.path.exists(file_path):
#                     # 文件存在，将当前行的数据（字典）加入valid_files列表
#                     valid_files.append(row)
        
#         # 按file_folder_id分组
#         grouped_files = group_files_by_field(valid_files)
        
#         return grouped_files
#     def group_files_by_field(files_list, group_field='file_folder_id'):
#         """
#         按指定字段分组文件（更通用的版本）
        
#         Args:
#             files_list: 包含文件信息字典的列表
#             group_field: 用于分组的字段名，默认为'file_folder_id'
        
#         Returns:
#             list: 分组后的文件列表
#         """
#         if not files_list:
#             return []
        
#         # 使用defaultdict按指定字段分组
#         grouped = defaultdict(list)
        
#         for file_info in files_list:
#             group_key = file_info.get(group_field)
#             if group_key is not None:  # 确保分组字段不为None
#                 grouped[group_key].append(file_info)
        
#         # 处理分组结果
#         result = []
        
#         for group_key, files in grouped.items():
#             if len(files) == 1:
#                 # 如果该分组只有一个文件，直接添加
#                 result.append(files[0])
#             else:
#                 # 如果有多个文件，第一个作为父元素，其余放在children中
#                 parent_file = files[0].copy()
#                 parent_file['children'] = files[1:]
#                 result.append(parent_file)
        
#         return result
#     _check_files_local_table()