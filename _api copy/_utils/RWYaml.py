import yaml
import os
from pathlib import Path

from ultralytics.utils import instance

def read_yaml(file_path):
    """
    读取YAML文件
    
    Args:
        file_path (str): YAML文件路径
    
    Returns:
        dict: YAML文件内容，如果文件不存在返回空字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data if data is not None else {}
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，将创建新文件")
        return {}
    except yaml.YAMLError as e:
        print(f"YAML文件解析错误: {e}")
        return {}

def write_yaml(file_path, data):
    """
    写入YAML文件
    
    Args:
        file_path (str): YAML文件路径
        data (dict): 要写入的数据
    """
    try:
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, default_flow_style=False, 
                     allow_unicode=True, sort_keys=False)
        print(f"成功写入文件: {file_path}")
    except Exception as e:
        print(f"写入文件时出错: {e}")

def _update_yaml_keys(file_path, new_path=None, new_train=None, new_val=None,new_names=None,delete=False):
    """
    更新YAML文件中的path、train、val键
    如果这些键存在，则删除重写；如果不存在，则直接添加
    
    Args:
        file_path (str): YAML文件路径
        new_path (str): 新的path值
        new_train (str): 新的train值  
        new_val (str): 新的val值
    """
    # 读取现有的YAML文件
    data = read_yaml(file_path)
    
    # 需要更新的键列表
    keys_to_update = ['path', 'train', 'val','names']
    new_values = [new_path, new_train, new_val,new_names]
    
    # 删除现有的path、train、val键（如果存在）
    if delete:
        for key in keys_to_update:
            if key in data and key!='names':
                del data[key]
                print(f"删除现有键: {key}")
    
    # 创建新的数据字典，将path、train、val放在最前面
    new_data = {}
    
    # 添加新的键值对（只添加非None的值）
    for key, value in zip(keys_to_update, new_values):
        if value is not None:
            new_data[key] = value
            print(f"添加新键: {key} = {value}")
    
    # 添加其他现有的键值对
    new_data.update(data)
    
    # 写入文件
    write_yaml(file_path, new_data)

def update_yaml_keys(file_path:str, ctx_list:list[dict]):
    """
    更新YAML文件中的path、train、val键
    如果这些键存在，则删除重写；如果不存在，则直接添加
    
    Args:
    path,train,val,names: 要更新的键值对
    ctx_list = [
        {'path':"new_value"}
    ]
    """
    # 读取现有的YAML文件
    new_data = {}
    data = read_yaml(file_path)
    for item in ctx_list:
        key,value = item.popitem()
        if value:
            del data[key]
            new_data[key] = value
    new_data.update(data)
    write_yaml(file_path, new_data)
            
            
    
    
    
    
    # # 需要更新的键列表
    # keys_to_update = ['path', 'train', 'val','names']
    # new_values = [new_path, new_train, new_val,new_names]
    
    # # 删除现有的path、train、val键（如果存在）
    # if delete:
    #     for key in keys_to_update:
    #         if key in data and key!='names':
    #             del data[key]
    #             print(f"删除现有键: {key}")
    
    # # 创建新的数据字典，将path、train、val放在最前面
    # new_data = {}
    
    # # 添加新的键值对（只添加非None的值）
    # for key, value in zip(keys_to_update, new_values):
    #     if value is not None:
    #         new_data[key] = value
    #         print(f"添加新键: {key} = {value}")
    
    # # 添加其他现有的键值对
    # new_data.update(data)
    
    # # 写入文件
    # write_yaml(file_path, new_data)


# 使用示例
if __name__ == "__main__":
    # 直接调用示例
    
    # 示例1：更新所有三个键
    # update_yaml_keys(
    #     file_path="config.yaml",
    #     new_path="datasets/my_dataset",
    #     new_train="images/train", 
    #     new_val="images/val"
    # )
    
    # # 示例2：只更新部分键
    # update_yaml_keys(
    #     file_path="config.yaml",
    #     new_path="new_dataset_path",
    #     new_train="new_train_path"
    #     # new_val不传参，表示不更新
    # )
    
    # # 示例3：读取文件内容
    # data = read_yaml("/home/panxiang/coding/kweilx/ultralytics/_api/data/dataset/yamls-202509021630-c251ba2d-f775-4231-9391-3914e258a200-dataset/data.yaml")
    # print("当前配置：", data)
    
    update_yaml_keys('/home/panxiang/coding/kweilx/ultralytics/_api/data/dataset/yamls-202509021700-bf7dc9c6-ad62-4ff0-b832-80e1b728d22c-dataset/data.yaml',
                     ctx_list=[{'path':"1"},{'train':"1"}])