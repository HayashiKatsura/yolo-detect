import os
import logging

# 获取当前项目所在路径
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import sys
sys.path.append(project_path)
from _api._utils.DataRecord import save_csv
from _api.configuration.FilesRecordMapping import table_1, table_2,table_3,table_4, table_5,files_record_mapping

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreateTable:
    """
    所有空能用到的数据表，这里为了简化操作，全部在本地创建实体csv表，并通过pandas读取
    实际生产环境中，建议使用数据库进行数据存储，这里只是为了方便测试和开发。
    """
    def __init__(self, delete_before_create=False, db_save_folder=os.path.join(project_path, '_api/data/table')):
        self.delete_before_create = delete_before_create
        self.db_save_folder = db_save_folder
        if os.path.exists(folder:=os.path.join(project_path, '_api/data/temp')):
            shutil.rmtree(folder)
        if self.delete_before_create:
            if os.path.exists(self.db_save_folder):
                shutil.rmtree(self.db_save_folder) 
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/cameras')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/images')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/datasets')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/weights')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/others')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/predict')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/train')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/val')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/dataset')):
                shutil.rmtree(folder)
            if os.path.exists(folder:=os.path.join(project_path, '_api/data/videos')):
                shutil.rmtree(folder)
            
            
            
        os.makedirs(db_save_folder, exist_ok=True)

    
    def _create_tables(self):
        """
        创建所有表格
        """
        self._camera_collect()
        self._weights_val()
        self._datasets()
        self._upload_images()
        self._other_files()
        self._detect_informations()
        self._train_val_folder()
        logger.info("数据表初始化。。。")
            
    def _camera_collect(self):
        """
        camera_collect表，前端通过摄像头采集的图像信息
        """
        save_csv(None, os.path.join(self.db_save_folder, 'camera_collections.csv'), table_1())
    
    def _weights_val(self):
        """
        db_save_folder表，所有训练的，或是上传的权重验证信息
        """
        save_csv(None, os.path.join(self.db_save_folder, 'weights_val.csv'), table_5())
    
    def _datasets(self):
        """
        datasets表，所有训练的，或是上传的数据集信息
        """
        save_csv(None, os.path.join(self.db_save_folder, 'datasets.csv'), table_3())
    
    def _upload_images(self):
        """
        upload_images表，所有上传的图片信息
        """
        save_csv(None, os.path.join(self.db_save_folder, 'upload_images.csv'), table_1())
        
    
    def _other_files(self):
        """
        其他文件表，如：训练好的模型，预测结果等
        """
        save_csv(None, os.path.join(self.db_save_folder, 'other_files.csv'), table_1())
        
    def _detect_informations(self):
        """
        检测信息表
        """
        save_csv(None, os.path.join(self.db_save_folder, 'detect_informations.csv'), table_2())
    
    def _train_val_folder(self):
        """
        训练集/验证集信息表
        """
        save_csv(None, os.path.join(self.db_save_folder, 'train_val.csv'), table_4())
    
    

    
