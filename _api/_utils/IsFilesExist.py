import os
pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(pj_folder)
from  _api._utils.ReadCSV import csv_key_to_value
from typing import NamedTuple
import pandas as pd

class FilesID(NamedTuple):
    param_name:str # 参数名
    file_id:str # 文件id
    

def is_files_exist(files_id:FilesID|list[FilesID]):

    """
    判断文件是否存在
    Args:
        files_id (dict | list): 文件id或文件id列表
        files_id[0] = {
            'param_name': 变量名,
            'file_id': 文件id,
        }
    Returns:
    {
        'index': 索引
        'file_id':  文件id
        'file_path':  文件路径
        'is_exist':  是否存在
        'file_type':  文件类型
    }
    """
    results = []
    files_id = [files_id] if isinstance(files_id,FilesID) else files_id
    for item in files_id:
        param_name,file_id = item
        data = []
        index = 0 # 索引
        invalid = 0 # 无效文件
        if not file_id:
            results.append(
                {
                    'param_name': param_name,
                    'data': data
                }
            )
            continue
        # 确保file_id为列表
        if not isinstance(file_id,list):
            try:
                file_id = file_id.split(',')
            except:
                file_id = [file_id] if isinstance(file_id,str) else file_id 
        for _file_id in file_id:
            file_type = None
            is_exist = False
            file_path = None
            real_name = None
            if not _file_id:
                invalid += 1
                is_exist = False
            else:
                file_path = csv_key_to_value(
                    source = {
                        'key':'file_id', 
                        'value': str(_file_id),
                        'csv': str(os.path.join(pj_folder, '_api/data/data_record.csv')),
                        },
                    target='file_path')
                create_time = csv_key_to_value(
                    source = {
                        'key':'file_id', 
                        'value': str(_file_id),
                        'csv': str(os.path.join(pj_folder, '_api/data/data_record.csv')),
                        },
                    target='file_create_time')
                if file_path:
                    if os.path.exists(file_path):
                        is_exist = True
                        if os.path.isfile(file_path):
                            file_type = 'file'
                        else:
                            file_type = 'folder'
                        real_name = os.path.basename(file_path)
                    else:
                        invalid += 1
                        is_exist = False
                else:
                    invalid += 1
                    is_exist = False
                
            data.append({
                'index': index,
                'file_id': _file_id,
                'file_path': file_path,
                'file_real_name': real_name,
                'is_exist': is_exist,
                'file_type': file_type,
                'file_create_time': create_time
            })
            index += 1
        if invalid == len(file_id):
            data = []
        results.append(
            {
                'param_name': param_name,
                'data': data
            }
        )
        
    return results




# def files_info(field_name: str, field_value, update_field: str = None, new_value = None) -> dict:
#     """根据字段查找行数据并修改指定字段的数据"""
#     result = None
#     record_path = str(os.path.join(pj_folder, '_api/data/data_record.csv'))
    
#     try:
#         df = pd.read_csv(record_path)
#         # 根据字段值查找行
#         matched_row = df[df[field_name] == field_value]
#     except Exception as e:
#         print(f"Error reading CSV: {e}")
#         return result
    
#     if matched_row.empty:
#         return result
    
#     # 如果传入了要修改的字段和新值，进行修改
#     if update_field and new_value is not None:
#         df.loc[df[field_name] == field_value, update_field] = new_value
#         try:
#             # 保存修改后的 DataFrame 回 CSV
#             df.to_csv(record_path, index=False)
#             print(f"Updated {update_field} to {new_value} for {field_name} = {field_value}")
#         except Exception as e:
#             print(f"Error saving updated CSV: {e}")
    
#     # 返回修改后的行数据
#     result = df[df[field_name] == field_value].iloc[0].to_dict()
#     return result

def files_info(field_name: str, field_value, update_field: str = None, new_value = None, delete_row: bool = False,record_path=None) -> dict:
    """根据字段查找行数据并修改指定字段的数据，或删除行数据"""
    result = None
    record_path = record_path if record_path else str(os.path.join(pj_folder, '_api/data/data_record.csv'))
    
    try:
        df = pd.read_csv(record_path)
        # 根据字段值查找行
        matched_row = df[df[field_name] == field_value]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return result
    
    if matched_row.empty:
        return result
    
    # 如果要删除行，删除后直接返回（无返回值）
    if delete_row:
        df_updated = df[df[field_name] != field_value]
        try:
            df_updated.to_csv(record_path, index=False)
            print(f"Deleted row where {field_name} = {field_value}")
        except Exception as e:
            print(f"Error saving CSV after deletion: {e}")
        return  # 无返回值
    
    # 如果传入了要修改的字段和新值，进行修改
    if update_field and new_value is not None:
        df.loc[df[field_name] == field_value, update_field] = new_value
        try:
            # 保存修改后的 DataFrame 回 CSV
            df.to_csv(record_path, index=False)
            print(f"Updated {update_field} to {new_value} for {field_name} = {field_value}")
        except Exception as e:
            print(f"Error saving updated CSV: {e}")
    
    # 返回修改后的行数据
    result = df[df[field_name] == field_value].iloc[0].to_dict()
    return result


    
    
    
                
if __name__ == '__main__':
    import pprint
    field_name = "file_id"
    field_value = "weight-202508141502-0f134886-2880-4d43-9043-edca977a0ac51"
    pprint.pprint(files_info(field_name, field_value))
    
