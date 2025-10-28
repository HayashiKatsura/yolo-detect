import os
pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(pj_folder)
from _api._utils.IsFilesExist import FilesID,is_files_exist
from _api._utils.DataRecord import data_record
from _api._utils.UnZip import extract_zip, extract_rar, extract_7z
from uuid import uuid4
import time
import shutil
import tempfile



def upload_files_scripts(folder_id,files_type,files_data, files_comment):
    results_files = []
    time_stamp = str(time.strftime('%Y%m%d%H%M', time.localtime()))
    folder_id = FilesID('folder_id',folder_id)
    if folder_id[1]:
        folder_info = is_files_exist(folder_id)[0]
        if folder_info['data'][0]['is_exist']:
            save_folder = folder_info['data'][0]['file_path']
            folder_id = folder_info['data'][0]['file_id']
    else:
        save_folder = str(os.path.join(pj_folder, '_api','data', '_NAME_', f"{time_stamp}-{str(uuid4())[20:]}"))  
    
    if files_type in [0,2]: # -> 图像文件 或 .pt权重文件
        _middle = 'images' if files_type == 0 else 'weights'
        save_folder = save_folder.replace('_NAME_', _middle) if not folder_id[1] else save_folder # 如果没有指定文件夹，则创建文件夹
        os.makedirs(save_folder, exist_ok=True)
        folder_id = f"folder-{time_stamp}-{str(uuid4())}" if not folder_id[1] else folder_id
        box_data = []
        for file in files_data:
            file_name = file.filename
            save_path = os.path.join(save_folder, file_name)
            file.save(save_path)
            record_msg = \
            {
                'file_id': f'{_middle}-{time_stamp}-{str(uuid4())}',
                'file_real_name': str(file_name),
                'file_type': f'{_middle}-file',
                'file_path':str(save_path),
                'file_comment': f'{_middle}-file-{files_comment}',
                'file_create_time':time_stamp 
            }
            data_record(record_msg)
            box_data.append({
                'file_id': record_msg['file_id'], # 保存的文件id
                'file_real_name': record_msg['file_real_name'],
                'file_type': record_msg['file_type'],
                'file_create_time':time_stamp,
                'file_type':'file' 
            })
        results_files.append({
            'folder_id': folder_id, # 保存的文件夹id
            'file_type':'file',
            'box_data': box_data
        })
    else: # -> 压缩包文件 压缩包里可能是图像文件、.pt权重文件
        folder_id = f"folder-{time_stamp}-{str(uuid4())}" if not folder_id[1] else folder_id
        for file in files_data:
            box_data = []
            file_name = file.filename
            file_ext = os.path.splitext(file_name)[1]
            # 创建临时文件
            temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
            try:
                # 关闭文件描述符，将使用文件名
                os.close(temp_fd)
                # 保存上传的文件到临时位置
                file.save(temp_path)
                temp_save_folder = str(uuid4())
                os.makedirs(temp_save_folder, exist_ok=True)
                # 根据文件扩展名选择不同的解压方法
                if file_ext == '.zip':
                    extract_zip(temp_path, temp_save_folder)
                elif file_ext == '.rar':
                    extract_rar(temp_path, temp_save_folder)
                elif file_ext == '.7z':
                    extract_7z(temp_path, temp_save_folder)
            
            except Exception as e:
                results_files.append({
                    'archive_name': str(file_name),
                    'file_type':'archive',
                    'box_data': box_data,
                    'error_msg': f'提取失败，{str(e)}'
                })
                continue
    
            finally:
                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # 确保只会创建一次文件夹
            for _f in os.listdir(temp_save_folder):
                _middle = 'images' if str(_f).lower().endswith(('.jpg', '.jpeg', '.png')) else 'weights'
                save_folder = save_folder.replace('_NAME_', _middle)
                os.makedirs(save_folder, exist_ok=True)
                # 移动文件到x新的位置
                shutil.move(os.path.join(temp_save_folder, _f), os.path.join(save_folder, _f))
                record_msg = \
                    {
                        'file_id': f"{_middle}-{time_stamp}-{str(uuid4())}",
                        'file_real_name': str(_f),
                        'file_type': f'{_middle}-file',
                        'file_path':str(os.path.join(save_folder, _f)),
                        'file_comment': f'{_middle}-file-{files_comment}',
                        'file_create_time': time_stamp 
                    }
                data_record(record_msg)
                box_data.append({
                    'file_id': record_msg['file_id'],
                    'file_real_name': record_msg['file_real_name'],
                    'file_type': record_msg['file_type'],
                    'file_create_time':time_stamp,
                    'file_type':'file' 
                })  
            results_files.append({
                'archive_name': str(file_name),
                'folder_id': folder_id, # 保存的文件夹id
                'file_type':'archive',
                'box_data': box_data
            })
            # 删除旧的位置
            shutil.rmtree(temp_save_folder)
    
    foler_record_msg = \
    {
        'file_id': folder_id,
        'file_real_name': os.path.basename(save_folder),
        'file_type': f'{_middle}-folder',
        'file_path':str(save_folder),
        'file_comment': f'{_middle}-folder',
        'file_create_time':time_stamp 
    }
    data_record(foler_record_msg)
    
    return results_files, folder_id

