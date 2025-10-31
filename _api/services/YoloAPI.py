import os
# 获取当前项目所在路径
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import sys
sys.path.append(PJ_ROOT)

from ultralytics import YOLO
from _api._utils.ReadCSV import csv_key_to_value
from _api._utils.DataRecord import data_record
from _api._utils.IsFilesExist import FilesID,is_files_exist,files_info
from _api._utils.UnZip import extract_zip, extract_rar, extract_7z
from _api._utils.UpdateYaml import update_yaml_config
from _api._utils.ReadCSV import get_line_data
from _api._utils.ImagestransferComponent.FromLocalImageFiles import TransferLocalImageFiles
from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType,table_1, table_2
from uuid import uuid4
import os
import pandas as pd
import torch
import gc
import time
import json
import tempfile
import glob
import cv2
import glob
import csv

from _project.mydata._code.yolo._test import standard_test,standard_test_video
from _project.mydata._code.yolo._val import *
from _project.mydata._code.yolo._train import standard_train

from _project.mydata._code._utils.files_process.videos_process import get_videos_frame_count

from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error
import multiprocessing
from sqlmodel import select
from _api.models.file import File as FileObject
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from sqlalchemy.orm import Session
from fastapi import APIRouter, Query, HTTPException
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
from typing import List, Dict, Any, Optional
from _api.configuration.RedisConfig import redis_config
import json
from datetime import datetime
import psutil


class YoloAPI:
    def __init__(self):
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        self.random_id = str(uuid4().hex)
        self.max_workers = 4
        self.redis_client = redis_config.get_client()

        
    def _get_storage_paths(self, weights_ids: List[int]) -> Dict[int, str]:
        """
        根据传入的 weights_ids 查询数据库，返回 id -> 存储路径 映射
        """
        with Session(SQL_ENGINE) as session:
            # 查询多个文件的 storage_path
            query = select(FileObject.id, FileObject.storage_path).where(FileObject.id.in_(weights_ids))
            result = session.execute(query).fetchall()

        # 将结果转换为 id -> path 的字典
        return {row[0]: row[1] for row in result}
        
    
    def exec_startTraining(self,train_params):
        """
        启动子线程
        """
        save_path = os.path.join(PJ_ROOT,'_api/data/train',f"{str(time.strftime('%Y%m%d%H%M', time.localtime()))}-{train_params['name']}")    
        try:
            process = multiprocessing.Process(
                target=standard_train, 
                kwargs={
                    'web_params': train_params,
                    'save_path': save_path
                }
            )
            process.start()
            msg = 'ok'
            code = 200
        except Exception as e :
            msg = str(e)
            code = 500
            save_path = None
        
        self.redis_client.setex(f"train-task:{datetime.now().date()}:{process.pid}", 3600*2, save_path)    
        train_params.update(
            {
                'process_id':process.pid,
                'msg':msg,
                'save_path':save_path,
                'code':code
            }
        )
        return train_params
        
    def exec_stopTraining(self,task_id):
        pid = int(task_id)
        code = 500
        try:
            process = psutil.Process(pid)
            info = {
                "pid": pid,
                "name": process.name(),
                "status": process.status(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }

            # 终止进程
            process.terminate()
            try:
                process.wait(timeout=5)  # 等待最多5秒
            except psutil.TimeoutExpired:
                process.kill()  # 强制结束

            code = 200
            return {"msg": 'ok', "info": info, "code":code}
        
        except psutil.NoSuchProcess:
            return {"msg": f"无{task_id}训练任务", "code":404}
        except psutil.AccessDenied:
            return {"msg": f"终止{task_id}时，权限域被拒绝", "code":code}
        except Exception as e:
            return {"msg": str(e), "code":code}
        
    def exec_showTraining(self, task_id: str, line_no: Optional[int] = None) -> Dict[str, Any]:
        """
        读取训练日志CSV并分页返回：
        - 未传 line_no：返回全部
        - 传了 line_no：返回从 line_no+1 到末尾
        - 若 line_no >= 最大行索引：仍返回最后一行，并删除 Redis key（仅触发一次）
        """
        redis_key = f"train-task:{datetime.now().date()}:{task_id}"
        save_path = self.redis_client.get(redis_key)
        if not save_path:
            return {"code": 404, "msg": f"无{task_id}训练任务"}

        if isinstance(save_path, str) and len(save_path) >= 2 and save_path[0] == save_path[-1] and save_path[0] in ('"', "'"):
            save_path = save_path[1:-1]

        log_path = os.path.join(save_path, "train", "results.csv")
        if not os.path.exists(log_path):
            return {"code": 404, "msg": f"无{task_id}训练任务，或训练日志未生成"}

        try:
            with open(log_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                all_rows: List[Dict[str, Any]] = list(reader)

            total_lines = len(all_rows)
            if total_lines == 0:
                return {"code": 204, "msg": "训练日志未生成", "task_id": task_id}

            # 计算起始行
            if line_no is None:
                start_idx = 0
            else:
                if line_no < 0:
                    line_no = 0
                # 常规情况：从 line_no+1 开始
                start_idx = line_no + 1
                # 边界：如果已经到达或超过最后一行索引，则只返回最后一行
                if start_idx >= total_lines:
                    start_idx = total_lines - 1
                    # 删除 Redis key（只会在到达末尾时触发一次）
                    self.redis_client.delete(redis_key)

            data = all_rows[start_idx:]  # 当到达末尾时，这里只会是最后一行

            return {
                "code": 200,
                "msg": "ok" if start_idx < total_lines - 1 else "reached end, key deleted",
                "task_id": task_id,
                "total_lines": total_lines,
                "returned_lines": len(data),
                "start_line": start_idx, 
                "data": data
            }

        except Exception as e:
            return {"code": 500, "msg": str(e), "task_id": task_id}

    

    def exec_validation(self, dataset_id:int,conf:str|float,weights_ids: List[int]) -> list:
        """
        并行验证多个权重文件，返回每个文件的验证结果
        - 先查询数据库获取存储路径
        - 使用线程池并发验证多个文件
        """
        # 获取存储路径（通过一次 IN 查询）
        with Session(SQL_ENGINE) as session:
            weights_query = select(FileObject.id, FileObject.storage_path).where(FileObject.id.in_(weights_ids))
            yaml_query = select(FileObject.id, FileObject.remark['yaml_path']).where(FileObject.id == dataset_id)
            weights_result = session.execute(weights_query).fetchall()
            yaml_result = session.execute(yaml_query).first()

        # 将查询结果转换为字典：{weight_id -> storage_path}
        weight_paths = {row[0]: row[1] for row in weights_result if row[0]!=dataset_id}

        # 验证文件
        results = []  # 用于存储验证结果
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(weight_paths))) as executor:
            futures = {}
            
            for weight_id, weight_path in weight_paths.items():
                # 定义保存路径
                save_folder = os.path.join(PJ_ROOT, '_api/data/validations', f'{self.time_stamp}-{weight_id}')
                
                # 创建文件夹（主线程中创建，避免并发问题）
                os.makedirs(save_folder, exist_ok=True)
                
                # 提交任务
                future = executor.submit(
                    standard_val,
                    val_data = {
                        'model': weight_path,
                        'yaml_data': yaml_result[1],
                        'weight_id': weight_id,
                        'conf_list':[float(conf)]
                    },
                    save_folder=save_folder
                )
                futures[future] = weight_id
            
            # 收集结果
            for future in as_completed(futures):
                weight_id = futures[future]
                try:
                    result = future.result()
                    results.append({
                        'weight_id': weight_id,
                        'result': result,
                        'msg': 'ok'
                    })
                                        
                    self.redis_client.setex(f"validation:weight-id:{weight_id}", 3600*2, json.dumps(result))    
                        
                except Exception as e:
                    print(f"Weight {weight_id} 执行失败: {e}")
                    results.append({
                        'weight_id': weight_id,
                        'msg': f'failed,{str(e)}'
                    })

            

        return results

    def exec_prediction(self, weight_id:int,files_ids: List[int]) -> list:
        """
        """
        MAX_BATCH = 100
        # 获取存储路径（通过一次 IN 查询）
        with Session(SQL_ENGINE) as session:
            files_query = select(FileObject.id, FileObject.storage_path).where(FileObject.id.in_(files_ids))
            weight_query = select(FileObject.id, FileObject.storage_path).where(FileObject.id == weight_id)
            files_result = session.execute(files_query).fetchall()
            weight_result = session.execute(weight_query).first()

        files_paths = {row[0]: row[1] for row in files_result}

        # 预测文件
        results = []  # 用于存储验证结果
        if len(files_paths)<=MAX_BATCH:
            results = standard_test(
                model_path=weight_result[1],
                test_image_path = files_paths
            )
        else:
            batch_files = [list(files_paths.items())[i:i+MAX_BATCH] for i in range(0, len(files_paths), MAX_BATCH)]
        
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch_files))) as executor:
                futures = {}

                # 提交任务
                for idx, batch in enumerate(batch_files):
                    batch_paths = dict(batch)
                    future = executor.submit(
                        standard_test,
                        model_path=weight_result[1],
                        test_image_path=batch_paths
                    )
                    futures[future] = idx

                # 收集结果
                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        result = future.result()
                        results += result
                    except Exception as e:
                        print(f"Batch {batch_idx} 执行失败: {e}")
        
        self.redis_client.setex(f"prediction:{datetime.now().date()}", 3600*24, json.dumps(results))    
        return results
            
  