import os
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(project_path)
from collections import defaultdict
import csv
from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType

from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error
from _api._utils.PrepareJsonfy import prepare_jsonfy

class ReadLocalTable:
    def __init__(self,file_type):
        self.file_type = file_type
        self.data_table_1,self.data_table_2  = None, None
        if str(self.file_type).find(',')!=-1:
            self.file_type_1,self.file_type_2 = self.file_type.split(',')
        else:
            self.file_type_1,self.file_type_2  = self.file_type,None
        self.data_table_1 = self.get_data_table(self.file_type_1)
        self.data_table_2 = self.get_data_table(self.file_type_2)
        self.data_table = [self.data_table_1,self.data_table_2]
            
    @staticmethod
    def get_data_table(file_type=None):
        data_table = None
        if not file_type:
            return data_table
        if file_type.find('null')!=-1 or file_type.find(type:=str(FilesType.images))!=-1:
            # data_table = files_record_mapping()[type][1]
            data_table = FileTable.query.all()
            data_table = [item.__dict__ for item in data_table]
            data_table = [prepare_jsonfy(item) for item in data_table]
            
        elif file_type.find(type:=str(FilesType.weights))!=-1:
            # data_table = files_record_mapping()[type][1]
            data_table = WeightTable.query.all()
            data_table = [item.__dict__ for item in data_table]
            data_table = [prepare_jsonfy(item) for item in data_table]
            
        elif file_type.find(type:=str(FilesType.yamls))!=-1:
            # data_table = files_record_mapping()[type][1]
            data_table = DatasetTable.query.all()
            data_table = [item.__dict__ for item in data_table]
            data_table = [prepare_jsonfy(item) for item in data_table]
            
        elif file_type.find(type:=str(FilesType.cameras))!=-1:
            # data_table = files_record_mapping()[type][1]
            
            data_table = FileTable.query.all()
            data_table = [item.__dict__ for item in data_table]
            data_table = [prepare_jsonfy(item) for item in data_table]
            
        return data_table
        
 
    @staticmethod
    def group_files_by_field(files_list, group_field='folder_path'):
        """
        按指定字段分组文件（更通用的版本）
        
        Args:
            files_list: 包含文件信息字典的列表
            group_field: 用于分组的字段名，默认为'file_folder_id'
        
        Returns:
            list: 分组后的文件列表，如果字段不存在则返回原始列表
        """
        if not files_list:
            return []
        
        # 检查是否存在指定的分组字段
        # 检查第一个文件是否包含该字段，如果没有则不进行分组
        if group_field not in files_list[0]:
            return files_list
        
        # 使用defaultdict按指定字段分组
        grouped = defaultdict(list)
        
        for file_info in files_list:
            group_key = file_info.get(group_field)
            if group_key is not None:  # 确保分组字段不为None
                grouped[group_key].append(file_info)
        
        # 处理分组结果
        result = []
        
        for group_key, files in grouped.items():
            if len(files) == 1:
                # 如果该分组只有一个文件，直接添加
                result.append(files[0])
            else:
                # 如果有多个文件，第一个作为父元素，其余放在children中
                # parent_file = files[0].copy()
                parent_file = {
                        'file_id': str(os.path.basename(group_key)),
                        'file_name': f"{str(group_key)[-12:]}",
                        'type': 'folder',
                        'comment': str(group_key)[-12:],
                        'create_time': str((group_key).split('-')[2])
                }
                # parent_file['children'] = files[1:]
                parent_file['children'] = files
                result.append(parent_file)
        
        return result
    def show_files_local_table(self):
        """
        获取当前的文件信息并按file_folder_id分组
        """
        grouped_files = []
        for sub_table in self.data_table:
            if not sub_table:
                continue
            
            valid_files = []

            # # 打开CSV文件并读取数据
            # with open(sub_table, mode='r', encoding='utf-8') as file:
            #     reader = csv.DictReader(file)
                
            #     # 遍历每一行
            #     for row in reader:
            #         # 获取file_path
            #         file_path = row.get('file_path')
                    
            #         # 检查文件是否存在
            #         if os.path.exists(file_path):
            #             # 文件存在，将当前行的数据（字典）加入valid_files列表
            #             valid_files.append(row)
            for item in sub_table:
                file_path = item.get('file_path',None)
                if file_path and os.path.exists(file_path):
                    valid_files.append(item)
                    
            # 按file_folder_id分组
            grouped_files+=self.group_files_by_field(valid_files)
        
        return grouped_files

if __name__ == '__main__':
    print(ReadLocalTable().show_files_local_table())