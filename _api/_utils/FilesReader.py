import os
from _api._utils.IsFilesExist import FilesID,is_files_exist,files_info

class FilesReader:
    def __init__(self,file_id):
        self.file_id = file_id
        self.file_info = files_info("file_id",file_id)
        if not self.file_info:
            return None
        self.file_path = self.file_info.get('file_path')

    def read_yaml(self):
                # 读取文件原始内容
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    
    def read_txt(self):
        pass
    
    def read_weights(self):
        pass
    
    def read_json(self):
        pass
    
    def read_csv(self):
        pass
    
    def read_image(self):
        pass
    
    def read_compressed(self):
        pass
    
    def read_file(self):
        pass
    
if __name__ == '__main__':
    file_path = '/Users/katsura/Documents/code/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml'
    reader = FilesReader(file_path)
    content = reader.read_yaml()
    print(content)