from abc import ABC, abstractmethod
from uuid import uuid4
import os
import zipfile
import py7zr
import time

parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_file_types(images_data):
    """
    检查传入的数据中包含的文件类型，确保只存在一种文件类型：图像文件、压缩包或.pt权重文件。

    Args:
        images_data (list): 包含多个文件的数据列表，每个元素是一个文件对象。

    Returns:
        tuple: (bool, int 或 None)
            - 如果是图像文件，返回 (True, 0)
            - 如果是压缩包文件，返回 (True, 1)
            - 如果是.pt权重文件，返回 (True, 2)
            - 如果没有文件类型或有多种类型的文件，返回 (False, None)
    """
    has_image = False
    has_compressed = False
    has_pt = False
    
    for file in images_data:
        ext = str(file.filename).lower()

        # 检查是否是压缩包文件
        if ext.endswith(('.zip', '.7z', '.rar')):
            has_compressed = True
        # 检查是否是图像文件
        elif ext.endswith(('.jpg', '.jpeg', '.png')):
            has_image = True
        # 检查是否是.pt权重文件
        elif ext.endswith('.pt'):
            has_pt = True
    
    # 检查是否有多种类型的文件存在
    if (has_image and (has_compressed or has_pt)) or (has_compressed and has_pt):
        return False, None
    
    # 返回对应的文件类型
    if has_image:
        return True, 0
    elif has_compressed:
        return True, 1
    elif has_pt:
        return True, 2
    else:
        return False, None


def files_real_path(files_list):
    results = []
    invalid = 0
    for item in files_list:
        file_id = item['file_id']
        file_path = item['file_path']        
        if not file_path:
            results.append({
                'file_id': file_id,
                'msg' : '文件不存在'
            })
            invalid += 1
        else:
            results.append({
                'file_id': file_id,
                'file_real_name': item['file_real_name'],
                'file_path': file_path,
                'file_type': item['file_type'],
                'file_create_time': str(item['file_create_time']),
                'msg' : '文件已获取'
            })
    return results, len(results)-invalid






# 获取当前文件的绝对路径
cur_dir = os.path.dirname(os.path.abspath(__file__))

class GetFiles(ABC):
    def __init__(self, files):
        self.files = files

    @abstractmethod
    def get_files(self):
        pass


class GetWeights(GetFiles):
    """
    获取权重文件
    Args:
        GetFiles (_type_): _description_
    """
    def __init__(self, files):
        super().__init__(files)

    def get_files(self):
        pass
    
class GetImages(GetFiles):
    """
    获取图片文件
    Args:
        GetFiles (_type_): _description_
    """
    def __init__(self, files):
        super().__init__(files)
        self.files = files
        
        
    def get_files(self,save_path = None):
        _suffix = save_path or f'{str(time.strftime("%Y%m%d%H%M", time.localtime()))}-{str(uuid4())[20:]}'
        save_folder = os.path.join(parent_folder, 'data', 'images', _suffix)
        os.makedirs(save_folder, exist_ok=True)
        self.files.save(os.path.join(save_folder, self.files.filename))
        return str(save_folder),_suffix

# class GetZip(GetFiles):
#     """
#     获取zip文件
#     Args:
#         GetFiles (_type_): _description_
#     """
#     def __init__(self, files):
#         super().__init__(files)
#         self.files = files
        
#     def get_files(self):
        
#         package_ext = self.files.filename.split('.')[-1]
#         # 保存路径
#         save_folder = os.path.join(parent_folder, 'data', 'images', str(uuid4()))
#         os.makedirs(save_folder, exist_ok=True)    
#         if package_ext == 'zip':
#             extract_zip(self.files, save_folder)
            
#         else:
#             extract_7z(self.files, save_folder)
#         return str(save_folder)
