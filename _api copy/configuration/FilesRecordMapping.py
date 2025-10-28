import os
pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(pj_folder)
from enum import Enum

class FilesType(Enum):
    """
    文件类型
    1. images: 上传的图像
    2. cameras: 摄像头采集的图像
    3. weights: 训练的权重
    4. datasets: 数据集
    5. others: 其他文件
    6. compressed: 压缩文件
    7. yamls: 数据集yaml文件
    8. detect: 检测信息
    """
    images = 'images'
    videos = 'videos'
    cameras = 'cameras'
    weights = 'weights'
    datasets = 'datasets'
    others = 'others'
    compressed = 'compressed'
    yamls = 'yamls'
    detect = 'detect'
    train  = 'train'
    val    = 'val'
    

    def __str__(self):
        return str(self.value)



def table_1():
    """
    一般文件
    Returns:
        _type_: _description_
    """
    return ["file_id","file_folder_id","file_path","file_folder_path","file_real_name","file_type",'file_comment',"is_detected","file_create_time"]

def table_2():
    """
    检测信息
    Returns:
        _type_: _description_
    """
    return ["file_id","weight_id","is_detected","file_create_time","update_time"]

def table_3():
    """
    数据集信息
    Returns:
        _type_: _description_
    """
    return ["file_id","yaml_path","file_path","file_real_name","file_type",'file_comment',"images_labels_folder_id","file_create_time"]

def table_4():
    """
    训练集/验证集信息
    Returns:
        _type_: _description_
    """
    return ["file_id","images_folder","labels_folder","dataset_id","file_create_time"]

def table_5():
    """
    权重文件
    Returns:
        _type_: _description_
    """
    return ["file_id","file_folder_id","file_path","file_folder_path","file_real_name","file_type",'file_comment','dataset_id',"is_detected","session_id","file_create_time"]


def files_record_mapping():
    """
    上传文件类型与数据文件的对应关系，数据文件记录了所有当前类型的数据信息
    数据格式（文件类型，数据文件路径）
    """
    return {

            str(FilesType.images):(str(FilesType.images),str(os.path.join(pj_folder,'_api/data/table/upload_images.csv'))), # 上传的图像
            str(FilesType.cameras):(str(FilesType.cameras),str(os.path.join(pj_folder,'_api/data/table/camera_collections.csv'))), # 摄像头采集的图像
            str(FilesType.weights):(str(FilesType.weights),str(os.path.join(pj_folder,'_api/data/table/weights_val.csv'))), # 训练的权重
            str(FilesType.datasets):(str(FilesType.datasets),str(os.path.join(pj_folder,'_api/data/table/datasets.csv'))), # 数据集
            str(FilesType.others):(str(FilesType.others),str(os.path.join(pj_folder,'_api/data/table/other_files.csv'))), # 其他文件
            # str(FilesType.compressed):(str(FilesType.compressed),str(os.path.join(pj_folder,'_api/data/table/compressed_files.csv'))), # 压缩文件
            str(FilesType.yamls):(str(FilesType.yamls),str(os.path.join(pj_folder,'_api/data/table/datasets.csv'))), # 数据集,
            str(FilesType.detect):(str(FilesType.detect),str(os.path.join(pj_folder,'_api/data/table/detect_informations.csv'))) ,# 检测信息
            str(FilesType.train):(str(FilesType.train),str(os.path.join(pj_folder,'_api/data/table/train_val.csv'))), # 训练集
            str(FilesType.val):(str(FilesType.val),str(os.path.join(pj_folder,'_api/data/table/train_val.csv'))), # 验证集
            str(FilesType.videos):(str(FilesType.videos),str(os.path.join(pj_folder,'_api/data/table/upload_videos.csv'))) # 视频文件
            
        }

if __name__ == '__main__':
    print(files_record_mapping()[str(FilesType.images)])
    # print(FilesType.cameras)