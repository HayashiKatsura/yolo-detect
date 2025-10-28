import os
pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(pj_folder)
from  _api._utils.ReadCSV import csv_key_to_value
import time
from uuid import uuid4


class ParseSaveUrl:
    """
    解析保存路径
    """
    def __init__(self, save_url:str):
        self.save_url = save_url
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        

    def parse_local_path(self):
        """
        传入路径id -> 解析本地路径
        Returns:
            实际路径
        """
        file_type = '' # 文件类型
        is_exist = False # 是否存在
        file_path = '' # 实际路径
        real_name = '' # 文件名
        
        file_path = csv_key_to_value(
            source = {
                'key':'file_id', 
                'value': str(self.save_url),
                'csv': str(os.path.join(pj_folder, '_api/data/data_record.csv')),
                },
            target='file_path')
        if file_path:
            if os.path.exists(file_path):
                is_exist = True
                file_type = 'file' if os.path.isfile(file_path) else 'folder'
                real_name = os.path.basename(file_path)
            else:
                is_exist = False
        else:
            is_exist = False
   
        return {
            'file_id': self.save_url,
            'file_path': file_path,
            'file_real_name': real_name,
            'is_exist': is_exist,
            'file_type': file_type,
        }
        
    def generate_local_save_path(self):
        local_save_info = self.parse_local_path()
        save_url = local_save_info['file_path'] if local_save_info['is_exist'] else None
        if not save_url:
            folder_id = f'folder-{self.time_stamp}-{str(uuid4())}'
            save_folder = str(os.path.join(pj_folder, '_api','data', '_NAME_', f"_NAME_-{folder_id}"))
        else:
            save_folder = local_save_info['file_path']
            folder_id = local_save_info['file_id']
        return {
            'file_path': save_folder,
            'file_id': folder_id,
        }
        
    def parse_netdisk_url(self):
        pass
    
    def generate_netdisk_save_url(self):
        pass
    
    