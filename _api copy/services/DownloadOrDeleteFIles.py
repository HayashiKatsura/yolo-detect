import os
# 获取当前项目所在路径
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(project_path)

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

# 文件类型映射
MIME_TYPES = {
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.zip': 'application/zip',
    '.rar': 'application/x-rar-compressed',
    '.7z': 'application/x-7z-compressed',
    '.pt': 'application/octet-stream',  # PyTorch 模型
    '.pth': 'application/octet-stream',  # PyTorch 模型
    '.pkl': 'application/octet-stream',  # Pickle 文件
    '.yaml': 'text/yaml',
    '.yml': 'text/yaml',
    '.txt': 'text/plain',
    '.log': 'text/plain',
    '.pdf': 'application/pdf',
    '.doc': 'application/msword',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.xls': 'application/vnd.ms-excel',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.mp4': 'video/mp4',
}

class DowenloadOrDeleteFilesLocalStorage:
    def __init__(self,
                 file_id=None,
                 detect_id=None,
                 val=None,
                 camera=None,
                 dataset_example=None,
                 train_log = None,
                 seesion_id=None,
                 train_id=None,
                 is_detected=None,
                 only_video_csv = None
                 ):
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        
        self.file_id = file_id
        self.detect_id = detect_id
        self.val = val
        self.camera = camera # 是否是采集的图像
        self.dataset_example = dataset_example # 是否是下载数据集样本
        self.train_log = train_log # 是否下载训练日志
        self.seesion_id = seesion_id # 下载的会话id
        self.train_id = train_id # 下载的训练id
        self.is_detected = is_detected # 是否下载检测结果
        self.file_id_path_map = {} # 下载文件id和路径映射
        self.only_video_csv = only_video_csv # 视频相关
        
        self.file_id = self.train_id if self.train_id else self.file_id
        
        self.temp_folder = os.path.join(project_path, '_api/data/temp') # 存放临时文件
        os.makedirs(self.temp_folder, exist_ok=True)
        
        if self.dataset_example:
            self.file_path = os.path.join(project_path,'_api/data/dataset_example/dataset.zip')
        elif self.train_log:
            self.file_path = os.path.join(project_path,'_api/logs/train',f'{self.seesion_id}.log')
        else:
            self.file_path = self.get_download_file_path()
        
        if self.file_id.find('video')!=-1: # 视频相关
            self.file_path = DetectionTable.query.filter_by(file_id=self.file_id).first().details.get("detect_txt_path",None)
    
    @handle_db_error  
    def get_download_file_path(self):
        # 复合id
        if str(self.file_id).find(',')!=-1:
            temp_folder = os.path.join(self.temp_folder, f"temp-{self.time_stamp}-{str(uuid4())[20:]}")
            os.makedirs(temp_folder, exist_ok=True)
        
            file_id_list = self.file_id.split(',')
            for sub_id in file_id_list:
                file_type = str(FilesType.cameras) if self.camera else str(FilesType.images)
                # id_info = files_info("file_id", 
                #                     sub_id,
                #                     record_path=files_record_mapping()[file_type][1])
                # id_file_path = id_info['file_path']
                id_info = FileTable.query.filter_by(file_id=sub_id).first()
                id_file_path = id_info.file_path
                is_detected = id_info.is_detected
            
                # 时间戳防止重名文件
                file_name = os.path.splitext(os.path.basename(id_file_path))[0]
                shutil.copy(id_file_path,os.path.join(temp_folder,f'{file_name}-{self.time_stamp}-origin.png')) # 原图
                
                # detect_image_txt_map = id_info.get('is_detected',None)
                # if detect_image_txt_map:
                if is_detected:
                    _details = DetectionTable.query.filter_by(file_id=sub_id).first().details
                    
                    # detect_image_id,detect_txt_id = detect_image_txt_map.split(',')
                    # detect_info= ast.literal_eval(files_info("file_id", sub_id,record_path=files_record_mapping()[str(FilesType.detect)][1]).get('is_detected',None))
                    # detect_image_path,detect_txt_path = detect_info[detect_image_id],detect_info[detect_txt_id]
                    detect_image_path,detect_txt_path = _details['detect_image_path'],_details['detect_txt_path']
                    
                    shutil.copy(detect_image_path,os.path.join(temp_folder,f'{file_name}-{self.time_stamp}-detect.png')) # 检测图
                    shutil.copy(detect_txt_path,os.path.join(temp_folder,f'{file_name}-{self.time_stamp}-detect.txt')) # 检测标签
                    
            to_be_download_file_path = temp_folder
        
        else:
            # 单id
            if str(self.file_id).find('folder')!=-1: # ->文件夹
                if self.train_id: # 下载训练文件夹
                    to_be_download_file_path = WeightTable.query.filter_by(file_id=self.train_id).first().folder_path
                
                else:
                    file_type = str(FilesType.cameras) if self.camera else str(FilesType.images)
                    # to_be_download_file_path = files_info("file_folder_id",  
                    #                                       self.file_id,
                    #                                       record_path=files_record_mapping()[file_type][1]).get('file_folder_path',None)
                    
                    
                    
                    # folder_info = FileTable.query.with_entities(FileTable.folder_path).all()
                    # folder_info = [i[0] for i in folder_info]
                    # for item in folder_info:
                    #     if item.find(self.file_id)!=-1:
                    #         to_be_download_file_path = item
                    #         break
                    folder_info = FileTable.query.with_entities(FileTable.file_id,FileTable.file_path,FileTable.folder_path,FileTable.is_detected).all()
                    for item in folder_info:
                        _file_id, _file_path, _folder_path,_detected = item
                        if str(_folder_path).find(self.file_id)!=-1:
                            to_be_download_file_path = _folder_path
                            self.file_id_path_map.update(
                                {str(os.path.basename(_file_path)):(str(_file_id),str(_file_path))}
                            )
                                    
            else: # ->文件 
                if not self.seesion_id: # 下载训练文件
                    
                    if str(self.file_id).find(file_type:=str(FilesType.cameras))!=-1: # 摄像头采集的图像
                        # to_be_download_file_path = files_info("file_id", 
                        #                             self.file_id,
                        #                             record_path=files_record_mapping()[file_type][1]).get('file_path',None)
                        to_be_download_file_path = FileTable.query.filter_by(file_id=self.file_id).first().file_path
                    
                    elif str(self.file_id).find(file_type:=str(FilesType.images))!=-1: # 上传的图像
                        # to_be_download_file_path = files_info("file_id", 
                        #                             self.file_id,
                        #                             record_path=files_record_mapping()[file_type][1]).get('file_path',None)
                        
                        to_be_download_file_path = FileTable.query.filter_by(file_id=self.file_id).first().file_path
                        
                    
                    elif str(self.file_id).find(file_type:=str(FilesType.videos))!=-1: # 上传的视频
                        # to_be_download_file_path = files_info("file_id", 
                        #                             self.file_id,
                        #                             record_path=files_record_mapping()[file_type][1]).get('file_path',None)
                        
                        to_be_download_file_path = FileTable.query.filter_by(file_id=self.file_id).first().file_path
                    
                    elif str(self.file_id).find(file_type:=str(FilesType.weights))!=-1: # 上传的权重
                        if not self.val: # 只下载权重
                            # to_be_download_file_path = files_info("file_id", 
                            #                         self.file_id,
                            #                         record_path=files_record_mapping()[file_type][1]).get('file_path',None)
                            to_be_download_file_path = WeightTable.query.filter_by(file_id=self.file_id).first().file_path
                            
                        else: 
                            # 下载验证文件
                            # to_be_download_file_path = files_info("file_id", 
                            #                         self.file_id,
                            #                         record_path=files_record_mapping()[file_type][1]).get('is_detected',None)
                            to_be_download_file_path = WeightTable.query.filter_by(file_id=self.file_id).first().is_validated
                            
                    elif str(self.file_id).find('train')!=-1: # 下载训练文件
                                #  to_be_download_file_path = files_info("file_id", 
                                #                     self.file_id,
                                #                     record_path=files_record_mapping()[str(FilesType.weights)][1]).get('file_path',None)
                            to_be_download_file_path = WeightTable.query.filter_by(file_id=self.file_id).first().folder_path
                                
                else:
            
                    # to_be_download_file_path = files_info("session_id", 
                    #                                 self.seesion_id,
                    #                                 record_path=files_record_mapping()[str(FilesType.weights)][1]).get('file_path',None)
                    to_be_download_file_path = WeightTable.query.filter_by(session_id=self.seesion_id).first().folder_path
                    
                    
        if not to_be_download_file_path:  
            raise FileNotFoundError("文件不存在")
        
        return to_be_download_file_path
        

    def download_files(self):
        if os.path.isfile(self.file_path) or str(self.file_path).endswith('.log'): # -> 下载单个文件
            # 下载结果图，下载文件夹
            if self.detect_id:
                
                # 检测文件位置
                # detect_image,detect_txt = self.detect_id.split(',')
                # detect_image_path= files_info("file_id", detect_image).get('file_path',None)
                # detect_txt_path= files_info("file_id", detect_txt).get('file_path',None)
                
                detect_infomation = DetectionTable.query.filter(DetectionTable.file_id == self.file_id ).first().details
                detect_image_path = detect_infomation.get('detect_image_path',None)
                detect_txt_path = detect_infomation.get('detect_txt_path',None)
                
                temp_folder = os.path.join(project_path, 'data', 'temp',f"temp-{self.time_stamp}-{str(uuid4())[20:]}",os.path.basename(self.file_path).split('.')[0])
                os.makedirs(temp_folder, exist_ok=True)
                # 原图
                shutil.copy(self.file_path,os.path.join(temp_folder,f'origin-{os.path.basename(self.file_path)}'))
                # 检测图
                shutil.copy(detect_image_path,os.path.join(temp_folder,os.path.basename(detect_image_path)))
                # 检测标签
                shutil.copy(detect_txt_path,os.path.join(temp_folder,os.path.basename(detect_txt_path)))
                self.file_path = temp_folder
                return self.download_folder()
            
            if str(self.file_id).find('video')!=-1:
                if str(self.only_video_csv) == "true": # 仅下载视频的csv文件
                    # self.file_path = f"{str(self.file_path).replace('.mp4','.csv')}"
                    return self.download_file()
                else: # 下载文件夹
                    self.file_path = os.path.dirname(self.file_path)
                    return self.download_folder()
            
            else:
                # 不是下载结果图，就直接下载
                return self.download_file()
            
        elif os.path.isdir(self.file_path): # -> 下载文件夹，打包成 ZIP 文件
            temp_folder = os.path.join(self.temp_folder, f"temp-{self.time_stamp}-{str(uuid4())[20:]}")
            os.makedirs(temp_folder, exist_ok=True)
            if self.is_detected:
                for _files in os.listdir(self.file_path):
                    try:
                        _file_id,_file_path = self.file_id_path_map[_files]
                    except:
                        continue
                    _detections = DetectionTable.query.filter(DetectionTable.file_id == _file_id ).first().details
                    _detect_image_path = _detections.get('detect_image_path',None)
                    _detect_txt_path = _detections.get('detect_txt_path',None)
                    # 拷贝原图
                    shutil.copy(_file_path,os.path.join(temp_folder,f'origin-{_files}'))
                    # 拷贝检测图
                    shutil.copy(_detect_image_path,os.path.join(temp_folder,f'{str(os.path.basename(_detect_image_path))}'))
                    # 拷贝检测标签
                    shutil.copy(_detect_txt_path,os.path.join(temp_folder,f'{str(os.path.basename(_detect_txt_path))}'))
                    
                self.file_path = temp_folder
                return self.download_folder()
            else:
                return self.download_folder()
        else:
            raise ValueError("无效的文件路径")
        
        
    def download_file(self):
        _, ext = os.path.splitext(os.path.basename(self.file_path))
        ext = ext.lower()
        mimetype = MIME_TYPES.get(ext, 'application/octet-stream')
        return {
            "path_or_file": self.file_path,
            "mimetype": mimetype,
            "download_name": os.path.basename(self.file_path),
            "need_delete":False
        }

    def download_folder(self):
        folder_name = os.path.basename(self.file_path)
        zip_name = f"{folder_name}.zip"
            # 检查文件夹大小，决定使用内存还是临时文件
        folder_size = self.get_folder_size(self.file_path)
        
        if folder_size < 100 * 1024 * 1024:  # 小于 100MB 使用内存
            # 使用内存中的 ZIP（适合小文件夹）
            zip_buffer, zip_filename = self.create_zip_from_folder(
                folder_path=self.file_path, 
                zip_name=zip_name)
            
            return {
                "path_or_file": zip_buffer,
                "mimetype": 'application/zip',
                "download_name": zip_filename,
                "need_delete":True
            }
            
        else:
            # 使用临时文件（适合大文件夹）
            temp_zip_path = self.create_temp_zip_file(
                folder_path=self.file_path, 
                zip_name=zip_name)
            
            return {
                "path_or_file": temp_zip_path,
                "mimetype": 'application/zip',
                "download_name": zip_name,
                "need_delete":True
            }
        
      
    def delete_files(self):
        """
        还需要删除表记录，未实现
        """
        if os.path.isfile(self.file_path):
            os.remove(self.file_path)
        elif os.path.isdir(self.file_path):
            shutil.rmtree(self.file_path)
        else:
            pass
            
   
   
   
    @staticmethod
    def get_folder_size(folder_path):
        """
        计算文件夹大小
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, IOError):
                    continue
        return total_size
    
    @staticmethod
    def create_zip_from_folder(folder_path, zip_name=None):
        """
        将文件夹压缩为 ZIP 文件
        返回 ZIP 文件的二进制数据
        """
        if zip_name is None:
            zip_name = f"{os.path.basename(folder_path)}.zip"
        
        # 使用内存中的字节流创建 ZIP
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 计算相对路径，保持文件夹结构
                    arcname = os.path.relpath(file_path, folder_path)
                    zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        return zip_buffer, zip_name
    @staticmethod
    def create_temp_zip_file(folder_path, zip_name=None):
        """
        创建临时 ZIP 文件（适用于大文件夹）
        返回临时文件路径
        """
        if zip_name is None:
            zip_name = f"{os.path.basename(folder_path)}.zip"
        
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        temp_zip_path = os.path.join(temp_dir, zip_name)
        
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zip_file.write(file_path, arcname)
        
        return temp_zip_path