from enum import Enum
from genericpath import isdir, isfile
import os
pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(pj_folder)
import time
from uuid import uuid4
from _api._utils.DataRecord import data_record
from _api._utils.UnZip import extract_zip, extract_rar, extract_7z
from abc import ABC, abstractmethod
import tempfile
import shutil
import logging

from  _api._utils.ParseSaveUrl import ParseSaveUrl
from _api._utils.UpdateYaml import update_yaml_config
from _api._utils.IsFilesExist import FilesID,is_files_exist,files_info
from _api._utils.ArchiveExtractor import ArchiveExtractor,extract_archive
from _api._utils.RWYaml import update_yaml_keys,read_yaml
from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error
from _api._utils.RWYaml import read_yaml

from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType,table_1, table_2,table_3,table_4

from _project.mydata._code._utils.files_process.txt_process import remove_last_column_in_txt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class UploadFileType(Enum):
    """
    上传文件类型
    """
    image = 'image'
    weight = 'weight'
    compressed = 'compressed'
    dataset = 'dataset'

    def __str__(self):
        return self.value

    

class UploadFilesLocalStorage():
    """
    上传文件到本地存储
    """
    def __init__(self, files_data, folder_id:str=None,camera=None):
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        self.folder_id = folder_id 
        self.files_data = files_data # 二进制文件
        self.is_dataset = False # 是否是数据集
        
        if self.folder_id == 'null': # -> 随机生成文件夹名
            self.folder_id = f'folder-{self.time_stamp}-{str(uuid4())}'
            self.save_folder = str(os.path.join(pj_folder, '_api','data', '_NAME_', f"_NAME_-{self.folder_id}"))
        else: # -> 保存到指定文件夹
            # self.save_folder = files_info("file_folder_id",self.folder_id)['file_folder_path']
            if str(self.folder_id).lower().find('yml')!=-1 or str(self.folder_id).lower().find('yaml')!=-1:
                this_dataset = DatasetTable.query.filter_by(file_id=self.folder_id).first()
                self.save_folder = {
                     "image":this_dataset.images_folder,
                     "label":this_dataset.labels_folder,
                     "yaml":this_dataset.yaml_path,
                 }
                self.is_dataset = True
 
        self.images_info, self.images_save_folder = None, self.save_folder
        self.yamls_info, self.yamls_save_folder = None, self.save_folder
        self.compressed_info, self.compressed_save_folder = None, self.save_folder
        self.weights_info, self.weights_save_folder = None, self.save_folder
        self.videos_info, self.videos_save_folder = None, self.save_folder
        self.box_data = [] # 保存的文件信息
        self.camera = camera # 是否是摄像头采集
        
        self.temp_folder = os.path.join(pj_folder, '_api/data/temp') # 存放临时文件
        os.makedirs(self.temp_folder, exist_ok=True)
    
    def upload_datasets(self,file_data):
        filename,content_type = file_data.filename,file_data.content_type
        temp_folder = os.path.join(self.temp_folder, f"temp-{self.time_stamp}-{str(uuid4())[20:]}")
        os.makedirs(temp_folder, exist_ok=True)
        archive_path = os.path.join(temp_folder, file_data.filename) # 临时存储的位置
        
        dataset_id = f'yamls-{self.time_stamp}-{str(uuid4())}' if not self.is_dataset else self.folder_id
        unzip_save_folder = os.path.join(pj_folder,'_api/data/dataset',f'{dataset_id}-{os.path.splitext(filename)[0]}') # 解压的位置
        inner_unzip_save_folder = unzip_save_folder
        os.makedirs(unzip_save_folder, exist_ok=True)
        
        file_data.save(archive_path)
        
        extract_archive(archive_path,unzip_save_folder)
        
        # macos 压缩的文件，会再额外生成一层文件夹，并且会产生‘__MACOSX’文件夹和‘.DS_Store'文件，需要删除
        folder_count = 0
        for sub_file in os.listdir(unzip_save_folder):
            if os.path.isdir(dir:=os.path.join(unzip_save_folder, sub_file)):
                if str(sub_file).lower().find('macos')!=-1:
                    shutil.rmtree(os.path.join(unzip_save_folder, sub_file))
                else:
                    inner_unzip_save_folder = dir
                    folder_count += 1
            if os.path.isfile(file:=os.path.join(unzip_save_folder, sub_file)) and str(file).lower().find('.DS_Store')!=-1:
                os.remove(file)
        
        if folder_count == 1: # 有两层文件夹
            for sub_file in os.listdir(inner_unzip_save_folder):
                shutil.move(os.path.join(inner_unzip_save_folder, sub_file), os.path.join(unzip_save_folder, sub_file))
            shutil.rmtree(inner_unzip_save_folder)
                
        
        # 获取图片/标签/yaml配置文件
        extract_folder_list = {}
        for sub_file in os.listdir(unzip_save_folder):
            
            if os.path.isfile(yaml:=os.path.join(unzip_save_folder, sub_file)):
                if str(yaml).lower().endswith(('.yaml','.yml')):
                    extract_folder_list['yaml'] = yaml
                else:
                    os.remove(os.path.join(unzip_save_folder, sub_file))
            
            elif os.path.isdir(dir:=os.path.join(unzip_save_folder, sub_file)):
                if str(sub_file).lower().find('macos')!=-1:
                    shutil.rmtree(os.path.join(unzip_save_folder, sub_file))
                
                if str(sub_file).lower().find('images')!=-1:
                    extract_folder_list['images'] = dir
                elif str(sub_file).lower().find('labels')!=-1:
                    extract_folder_list['labels'] = dir
        
        yaml_path = extract_folder_list.get('yaml',None)
        images_path = extract_folder_list.get('images',None)
        labels_path = extract_folder_list.get('labels',None)
        
        if labels_path: # 格式化标签文件
            remove_last_column_in_txt(os.path.join(labels_path, 'train'))
            remove_last_column_in_txt(os.path.join(labels_path, 'val'))
            
        def get_count(folder_path):
            results = {}
            for item in ['train','val']:
                count = 0
                if os.path.isdir(this_folder:=os.path.join(folder_path, item)):
                    count += len([f for f in os.listdir(this_folder)if str(f).lower().endswith(('.jpg','.png','.jpeg'))])
                    results.update({item:count})
            return results
        counts = get_count(images_path)
        train_count, val_count = counts.get('train',0), counts.get('val',0)
        if not self.is_dataset: # 上传新的数据集
            # TODO 可能需要更严谨的验证方式
            if not (yaml_path and images_path and labels_path):
                raise Exception('数据集格式不正确')
            
            images_labels_folder_id = f'images-labels-folder-{self.time_stamp}-{str(uuid4())}'
            update_yaml_keys(
                            file_path=str(yaml_path),
                            ctx_list=[
                                {'path':str(unzip_save_folder)},
                                {'train':str(os.path.join(unzip_save_folder, "images/train"))},
                                {'val':str(os.path.join(unzip_save_folder, "images/val"))},
                            ] )
            
            record_msg = DatasetTable(
            file_id = dataset_id,
            file_path = str(unzip_save_folder),
            file_name = str(filename).split('.')[0],
            yaml_path = str(yaml_path),
            images_folder = str(os.path.join(unzip_save_folder, "images")),
            labels_folder = str(os.path.join(unzip_save_folder, "labels")),
            train_counts = train_count,
            last_train_counts = train_count,
            val_counts = val_count,
            last_val_counts = val_count,
            )
            db.session.add(record_msg)
            
        else: # 往原数据集里加数据
            target_images_folder,target_labels_folder,target_yaml_path = \
                self.save_folder.get('image',None),self.save_folder.get('label',None), self.save_folder.get('yaml',None) # 源数据集图像标签路径
            if not (target_images_folder and target_labels_folder):
                raise Exception('数据集格式不正确')
            
            if yaml_path: # 更新yaml配置文件
                new_names = read_yaml(yaml_path).get('names',None)
                new_names = None if new_names=={} else new_names
            
                update_yaml_keys(
                file_path=str(target_yaml_path),
                ctx_list=[{'names':new_names}])
            
            for item in ['train','val']:
                for _sub_folder in os.listdir(images_path):
                    if os.path.isdir(this_folder:=os.path.join(images_path, _sub_folder)) and _sub_folder.lower().find(item)!=-1: # 训练
                        for _sub_file in os.listdir(this_folder):
                            if str(_sub_file).lower().endswith(('.jpg', '.jpeg', '.png')):
                                label_file_name = f'{os.path.splitext(_sub_file)[0]}.txt'
                                if os.path.exists(label_file:=os.path.join(labels_path, _sub_folder,label_file_name)) and os.path.isfile(label_file):
                                    # 拷贝图像
                                    shutil.move(os.path.join(this_folder, _sub_file), 
                                                os.path.join(target_images_folder,item,f'{self.time_stamp}_{_sub_file}')) # 加时间戳防止重名
                                    # 拷贝标签
                                    shutil.move(label_file, 
                                                os.path.join(target_labels_folder,item, f'{self.time_stamp}_{label_file_name}'))
            shutil.rmtree(unzip_save_folder)
            dataset = DatasetTable.query.filter_by(file_id=self.folder_id).first()
            last_train_counts, last_val_counts = dataset.train_counts, dataset.val_counts
            dataset.update_time = self.time_stamp
            dataset.train_counts = train_count + last_train_counts
            dataset.last_train_counts = last_train_counts
            dataset.val_counts = val_count + last_val_counts
            dataset.last_val_counts = last_val_counts
        db.session.commit()
        
        # 记录数据集
        # record_msg = \
        # {
        #    "file_id":dataset_id,
        #    'yaml_path':str(yaml_path),
        #    "file_path":str(unzip_save_folder),
        #    "file_real_name":str(filename).split('.')[0],
        #    "file_type":'folder',
        #    'file_comment':'dataset-folder',
        #    'images_labels_folder_id':images_labels_folder_id,
        #    "file_create_time":self.time_stamp,
        # }
        # data_record(record_msg,table_3(),files_record_mapping()[str(FilesType.datasets)][1])

        
        # 记录训练集/验证集
        # record_msg = \
        # {
        #     "file_id":images_labels_folder_id,
        #     "images_folder":str(os.path.join(unzip_save_folder, "images")),
        #     "labels_folder":str(os.path.join(unzip_save_folder, "labels")),
        #     "dataset_id":dataset_id,
        #     "file_create_time":self.time_stamp,
        # }
        # data_record(record_msg,table_4(),files_record_mapping()[str(FilesType.train)][1])
        
    @staticmethod
    def get_real_folder_name(folder_path, map=['train','val']):
        """
        获取文件夹的实际名称

        Args:
            folder_path (_type_): _description_
            map (list, optional): _description_. Defaults to ['train','val'].
            map=['image','label']
        Returns:
            返回实际路径
        """
        result = {}
        for sub_folder in os.listdir(folder_path):
            if os.path.isdir(this_folder:=os.path.join(folder_path, sub_folder)):
                if sub_folder.lower().find(map[0])!=-1: # 训练
                    result.update({map[0]:this_folder})
                elif sub_folder.lower().find(map[1])!=-1:
                    result.update({map[1]:this_folder})
        return result

    def _upload_file(self):
        for file_data in self.files_data:
            filename,content_type = file_data.filename,file_data.content_type
            # 压缩文件
            if str(filename).lower().endswith(('.zip', '.rar', '.7z')):
                self.upload_datasets(file_data)
                
            # 图像文件
            elif str(filename).lower().endswith(('.jpg', '.jpeg', '.png')):
                self.images_info, self.images_save_folder = self.upload_file(str(FilesType.images),
                                                                             file_data,
                                                                             self.images_save_folder)
                self.box_data.append(self.images_info)
            
            # 视频文件
            elif str(filename).lower().endswith(('.mp4', '.avi', '.mov')):
                self.videos_info, self.videos_save_folder = self.upload_file(str(FilesType.videos),
                                                                             file_data,
                                                                             self.videos_save_folder)
                self.box_data.append(self.videos_info)
            
            
            # 配置文件
            elif str(filename).lower().endswith(('.yaml','.yml')):
                self.yamls_info, self.yamls_save_folder = self.upload_file(str(FilesType.yamls),
                                                                           file_data,
                                                                           self.yamls_save_folder)
                self.box_data.append(self.yamls_info)
            # 权重文件
            elif str(filename).lower().endswith(('.pt')):
                self.weights_info, self.weights_save_folder = self.upload_file(str(FilesType.weights),
                                                                               file_data,
                                                                               self.weights_save_folder)
                self.box_data.append(self.weights_info)
            
          
            
            
            
            
                
        return self.box_data
    
    
    def upload_images(self,file_data,save_folder=None):
        """
        上传图像文件
        """
        file_name = file_data.filename
        if str(file_name).find('collect')!=-1: # 摄像头采集的图像
            save_folder = self.save_folder.replace('_NAME_', files_record_mapping()['collect'][0]) if self.save_folder.find('_NAME_') != -1 else self.save_folder
            db_save_path = files_record_mapping()['collect'][1]
        else: # 上传的图像
            save_folder = self.save_folder.replace('_NAME_', files_record_mapping()['images'][0]) if self.save_folder.find('_NAME_') != -1 else self.save_folder
            db_save_path = files_record_mapping()['images'][1]
        
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file_name)
        file_data.save(save_path)
        record_msg = \
        {
            'file_id': f'image-{self.time_stamp}-{str(uuid4())}',
            'file_folder_id': self.folder_id,
            'file_real_name': str(file_name),
            'file_type': 'image',
            'file_path':str(save_path),
            'file_folder_path':str(save_folder),
            'file_comment': 'upload_image',
            'file_create_time':self.time_stamp,
            'is_detected':False,
        }
        data_record(record_msg,db_save_path)
        return record_msg,str(save_folder)
    
    @handle_db_error
    def upload_file(self,file_type:str,file_data,save_folder:str=None):
        """
        上传单个文件
        """
        if file_data.filename.find(',')!=-1: # 同时存在文件id和文件名 （文件id,文件名）
            file_id,file_name = file_data.filename.split(',')
        else:
            file_id = f'{file_type}-{self.time_stamp}-{str(uuid4())}'
            file_name = file_data.filename
            
        file_comment_prefix = 'upload'
        
        if file_type == str(FilesType.images):
            if str(file_name).find('collect')!=-1: # 摄像头采集的图像
                save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.cameras)][0]) \
                    if self.save_folder.find('_NAME_') != -1 \
                    else self.save_folder
                db_save_path = files_record_mapping()[str(FilesType.cameras)][1]
                file_comment_prefix = 'camera'
                file_type = str(FilesType.cameras)
            else: # 上传的图像
                save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.images)][0]) \
                    if self.save_folder.find('_NAME_') != -1 \
                    else self.save_folder
                db_save_path = files_record_mapping()[str(FilesType.images)][1]
        
        elif file_type == str(FilesType.videos):
            # 上传的视频
            save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.videos)][0]) \
                if self.save_folder.find('_NAME_') != -1 \
                else self.save_folder
            db_save_path = files_record_mapping()[str(FilesType.videos)][1]
        
        
        elif file_type == str(FilesType.yamls):
                save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.datasets)][0]) \
                    if self.save_folder.find('_NAME_') != -1 \
                    else self.save_folder
                db_save_path = files_record_mapping()[str(FilesType.datasets)][1]
        
        elif file_type == str(FilesType.weights):
                save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.weights)][0]) \
                    if self.save_folder.find('_NAME_') != -1 \
                    else self.save_folder
                db_save_path = files_record_mapping()[str(FilesType.weights)][1]
        
        elif file_type == str(FilesType.datasets):
                save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.datasets)][0]) \
                    if self.save_folder.find('_NAME_') != -1 \
                    else self.save_folder
                db_save_path = files_record_mapping()[str(FilesType.datasets)][1]
        
        elif file_type == str(FilesType.others):
                save_folder = self.save_folder.replace('_NAME_', files_record_mapping()[str(FilesType.others)][0]) \
                    if self.save_folder.find('_NAME_') != -1 \
                    else self.save_folder
                db_save_path = files_record_mapping()[str(FilesType.others)][1]
        else:
            pass
        
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, file_name)
        file_data.save(save_path)
        # TODO 切换数据库
        # file_record = \
        # {
        #     'file_id': str(file_id),
        #     'file_folder_id': str(os.path.basename(save_folder)),
        #     'file_real_name': str(file_name),
        #     'file_type': f'{file_type}',
        #     'file_path':str(save_path),
        #     'file_folder_path':str(save_folder),
        #     'file_comment': f'{file_comment_prefix}_{file_type}',
        #     'file_create_time':self.time_stamp,
        #     'is_detected':False,
        # }
        
        # data_record(file_record,table_1(),db_save_path)
        
        if str(file_type).find('weight')!=-1:
            file_record = WeightTable(
                    file_id = str(file_id),
                    file_path = str(save_path),
                    folder_path = str(save_folder),
                    file_name = str(file_name)
            )
        elif str(file_type).find('image')!=-1:
            file_record =FileTable(
                    file_id = str(file_id),
                    file_path = str(save_path),
                    folder_path = str(save_folder),
                    file_name = str(file_name),
                    type = f'{file_type}',
                    comment = f'{file_comment_prefix}_{file_type}',
                    create_time = self.time_stamp
            )
            
        elif str(file_type).find('videos')!=-1:
            file_record =FileTable(
                    file_id = str(file_id),
                    file_path = str(save_path),
                    folder_path = str(save_folder),
                    file_name = str(file_name),
                    type = f'{file_type}',
                    comment = f'{file_comment_prefix}_{file_type}',
                    create_time = self.time_stamp
            )
            
        elif str(file_type).find('camera')!=-1:
            file_record =FileTable(
                    file_id = str(file_id),
                    file_path = str(save_path),
                    folder_path = str(save_folder),
                    file_name = str(file_name),
                    type = f'{file_type}',
                    comment = f'{file_comment_prefix}_images'
            )
        elif str(file_type).find('ml')!=-1:
            dataset_save_folder = read_yaml(str(save_path))['path']
            file_record =DatasetTable(
                    file_id = str(file_id),
                    file_path = str(save_path),
                    yaml_path = str(save_path),
                    file_name = str(file_name),
                    images_folder = str(os.path.join(dataset_save_folder, "images")),
                    labels_folder = str(os.path.join(dataset_save_folder, "lables"))
            )
            
        db.session.add(file_record)
        db.session.commit()
        
        return 'ok',str(save_folder)
        # return True
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    def _images(self):
        """
        上传图像文件
        """
        save_folder = self.save_folder.replace('_NAME_', 'images') if self.save_folder.find('_NAME_') != -1 else self.save_folder
        os.makedirs(save_folder, exist_ok=True)
        box_data = [] # 保存的文件信息
        results_files = [] # 返回的结果信息
        for _files in self.files_data:
            file_name = _files.filename
            if str(file_name).lower().endswith(('.jpg', '.jpeg', '.png')):
                save_path = os.path.join(save_folder, file_name)
                _files.save(save_path)
                record_msg = \
                {
                    'file_id': f'image-{self.time_stamp}-{str(uuid4())}',
                    'file_folder_id': self.folder_id,
                    'file_real_name': str(file_name),
                    'file_type': 'image',
                    'file_path':str(save_path),
                    'file_comment': 'image',
                    'file_create_time':self.time_stamp,
                }
                data_record(record_msg)
                box_data.append({
                    'file_id': record_msg['file_id'], # 保存的文件id
                    'file_real_name': record_msg['file_real_name'],
                    'file_type': record_msg['file_type'],
                    'file_create_time':record_msg['file_create_time'],
                    'file_type':record_msg['file_type'] 
                })
        results_files.append({
            'folder_id': self.folder_id, # 保存的文件夹id
            'file_type':'images',
            'box_data': box_data
        })

        return results_files
    
    def _weights(self):
        """
        上传模型权重
        """
        save_folder = self.save_folder.replace('_NAME_', 'weights') if self.save_folder.find('_NAME_') != -1 else self.save_folder
        os.makedirs(save_folder, exist_ok=True)
        box_data = [] # 保存的文件信息
        results_files = [] # 返回的结果信息
        for _files in self.files_data:
            file_name = _files.filename
            if str(file_name).lower().endswith(('.pt')):
                save_path = os.path.join(save_folder, file_name)
                _files.save(save_path)
                record_msg = \
                {
                    'file_id': f'weight-{self.time_stamp}-{str(uuid4())}',
                    'file_folder_id': self.folder_id,
                    'file_real_name': str(file_name),
                    'file_type': 'weight',
                    'file_path':str(save_path),
                    'file_comment': 'weight',
                    'file_create_time':self.time_stamp,
                }
                data_record(record_msg)
                box_data.append({
                    'file_id': record_msg['file_id'], # 保存的文件id
                    'file_real_name': record_msg['file_real_name'],
                    'file_type': record_msg['file_type'],
                    'file_create_time':record_msg['file_create_time'],
                    'file_type':record_msg['file_type'] 
                })
        results_files.append({
            'folder_id': self.folder_id, # 保存的文件夹id
            'file_type':'weights',
            'box_data': box_data
        })

        return results_files
    
    def _compressed(self):
        """
        上传压缩包
        """
        pass
    
    def _datasets(self):
        """
        上传数据集
        """
        import glob
        save_folder = self.save_folder.replace('_NAME_', 'datasets') if self.save_folder.find('_NAME_') != -1 else self.save_folder
        os.makedirs(save_folder, exist_ok=True)
        file_ext = os.path.splitext(self.files_data[0].filename)[-1].lower()
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
        try:
            # 关闭文件描述符，将使用文件名
            os.close(temp_fd)
            # 保存上传的文件到临时位置
            self.files_data[0].save(temp_path)
            os.makedirs(save_folder, exist_ok=True)
            # 根据文件扩展名选择不同的解压方法
            if file_ext == '.zip':
                extract_zip(temp_path, save_folder)
            elif file_ext == '.rar':
                extract_rar(temp_path, save_folder)
            elif file_ext == '.7z':
                extract_7z(temp_path, save_folder)
        
        except Exception as e:
            return f'Extraction failed: {str(e)}'
        
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        yaml_files = glob.glob(os.path.join(save_folder, '*.yaml'))
        update_yaml_config(
            yaml_file = yaml_files[0], 
            new_values = {
            "path": str(save_folder),
            "train": str(os.path.join(save_folder, 'images/train')),
            "val": str(os.path.join(save_folder, 'images/val'))
            }
        )
        #  yaml_data
        yaml_record_msg = \
                {
                    'file_id': f"yaml-{self.time_stamp}-{str(uuid4())}",
                    'file_real_name': str(os.path.basename(yaml_files[0])),
                    'file_type': 'yaml',
                    'file_path':str(yaml_files[0]),
                    'file_comment': 'yaml',
                    'file_create_time': str(self.time_stamp)
                }
        data_record(yaml_record_msg)
        
        # 训练文件夹
        datasets_folder = os.path.dirname(yaml_files[0])
        folder_list = []
        for _dir in ['images', 'labels']:
            for _condition in ['train', 'val']:
                _folder = f"{_dir}/{_condition}"
                _folder_path = os.path.join(datasets_folder, _folder)
                folder_record_msg = \
                    {
                        'file_id': f"folder-{self.time_stamp}-{str(uuid4())}",
                        'file_real_name': str(_condition),
                        'file_type': 'dataset-folder',
                        'file_path':str(_folder_path),
                        'file_comment': 'dataset-folder',
                        'file_create_time': str(self.time_stamp)
                    }
                data_record(folder_record_msg)
                folder_list.append({
                    'folder_name': str(_folder),
                    'folder_id': folder_record_msg['file_id'],
                })
        
        return {"yaml_file_id": yaml_record_msg['file_id'],"save_folder":folder_list}
        

class UploadFilesCloudOSS():
    """
    上传文件到云存储
    """
    def __init__(self, files_data, folder_id:str=None):
        super().__init__(files_data, folder_id)
        pass
    
    def _images(self):
        """
        上传图像文件
        """
        pass
    def _weights(self):
        """
        上传模型权重
        """
        pass
    def _compressed(self):
        """
        上传压缩包
        """
        pass
    def _datasets(self):
        """
        上传数据集
        """
        pass

