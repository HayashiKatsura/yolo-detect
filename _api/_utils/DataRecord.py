import os
import csv
import logging
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    

import os
import csv
from typing import List, Dict, Optional

def save_csv(data: Optional[List[Dict[str, str]]], save_path: str,fieldnames:List[str] =None) -> None:
    # 判断文件是否已存在
    file_exists = os.path.isfile(save_path)
    
    # 表头
    fieldnames = fieldnames if fieldnames else ['file_id','file_folder_id','file_real_name','file_type','file_path','file_folder_path','file_comment','file_create_time','is_detected']
    
    # 当传入的data为None时的处理逻辑
    if data is None:
        if file_exists:
            # 如果表格已存在，直接退出函数
            print(f"File {save_path} already exists, exiting function.")
            return
        else:
            # 如果表格不存在，只创建表格并写入表头
            with open(save_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            logger.info(f"{os.path.basename(save_path)}初始化完成")
            return
    
    # 处理数据，确保每条数据都包含表头的所有字段，缺失的字段填充为 None
    for row in data:
        for field in fieldnames:
            if field not in row:
                row[field] = 'None'
    
    # 写入 CSV 文件
    with open(save_path, mode='a', newline='') as csvfile: 
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 如果文件不存在,写入表头
        if not file_exists:
            writer.writeheader()
        # 写入数据
        writer.writerows(data)

    logger.info(f"{os.path.basename(save_path)}初始化完成")


def data_record(data: dict | list, fieldnames=None,save_path: str = None) -> None:
    # 获取当前文件的绝对路径
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = save_path or os.path.join(parent_dir, 'data', 'data_record.csv')
    
    # 确保数据是列表
    data = [data] if isinstance(data, dict) else data
    
     # 表头
    fieldnames = fieldnames if fieldnames else ['file_id','file_folder_id','file_real_name','file_type','file_path','file_folder_path','file_comment','file_create_time','is_detected']
    
    # 调用保存函数
    save_csv(data, save_path,fieldnames)