import os
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import sys
sys.path.append(PJ_ROOT)
from ultralytics import YOLO
from ultralytics import RTDETR

import time
import torch
import gc
import yaml
import os
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from _api.models.file import File as FileObject
import time
from _api.configuration.RedisConfig import redis_config
from sqlmodel import Session,select
import json


from loguru import logger as lgu
lgu.remove()
# 自定义日志格式（不包含 name:function:line）
lgu_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | " \
      "<level>{level: <8}</level> | " \
      "<cyan>{message}</cyan>"

def standard_train(config_yaml_path: str = None,
                   data_yaml: str = None ,
                   weight_path: str= None,
                   desc: str ='',
                   save_path:str = None,
                   batch_size:int = 8,
                   epochs:int = 1,
                   image_size:int = 640,
                   learning_rate:float = 0.01,
                   device:int|str = 0,
                   freeze:list = None,
                   rtd_yolo:str = 'yolo',
                   web_params:dict = None)->str:
    """
    默认跑在yolov8n上

    Args:
        config_yaml_path (str): 模型配置文件路径
        data_yaml (str): 数据集配置文件路径
        weight_path (str, optional): 模型权重文件路径. Defaults to None.
        pr_name (str, optional): 训练结果保存的文件夹名称. Defaults to ''.
        desc (str, optional): 训练结果保存的文件夹描述. Defaults to ''.
        save_path (str, optional): 训练结果保存的文件夹路径. Defaults
        
        batch_size (int, optional): 批次大小. Defaults to 8.
        epochs (int, optional): 训练轮数. Defaults to 500.
        image_size (int, optional): 训练图片尺寸. Defaults to 640.
        learning_rate (float, optional): 学习率. Defaults to 0.01.
        device (int|str, optional): 训练设备. Defaults to 0.
        freeze (list, optional): 冻结层. Defaults to None.
        
        web_params (dict, optional): web端训练参数. Defaults to None.
        session_id (str, optional): 训练会话ID. Defaults to '123'.
        
        
    Returns:
        str: 训练结果保存路径
    """
    time_stamp = str(time.strftime('%Y%m%d%H%M', time.localtime()))
    if web_params: # for web
        model_path_map = {
            "yolov8": str(os.path.join(PJ_ROOT,"ultralytics/cfg/models/8/yolov8.yaml")),
            "yolov11": str(os.path.join(PJ_ROOT,"ultralytics/cfg/models/11/yolo11.yaml")),
            "yolov12": str(os.path.join(PJ_ROOT,"ultralytics/cfg/models/12/yolo12.yaml")),
            "chipsyolo": str(os.path.join(PJ_ROOT,"ultralytics/cfg/models/12/yolo12.yaml")),
        }
        
        dataset_id = int(web_params['dataset_id'])
        train_name =  web_params.get("name",f"{time_stamp}")
        modelVersion = web_params.get('model','yolov8').lower()
        rtd_yolo = 'yolo' if str(modelVersion).find('yolo')!=-1 else 'rtdetr'
        device = (f"cuda:0" if torch.cuda.is_available() else "cpu") or (f"cuda:0" if str(web_params.get('device','cpu')).lower() == "gpu" else "cpu")
        image_size = int(web_params.get('image_size',640))
        batch_size =int( web_params.get('batch_size',8))
        learning_rate = float(web_params.get('learning_rate',0.01))
        epochs = int(web_params.get('epochs',1))
        session_id = f"{os.path.basename(save_path)}"
        
        with Session(SQL_ENGINE) as session:
            query = select(FileObject).where(FileObject.id == dataset_id)
            dataset_data = session.exec(query).first()
            if not dataset_data:
                raise ValueError(f"数据集文件ID {dataset_id} 不存在")
            dataset_data = dataset_data.remark
        data_yaml = dataset_data['yaml_path']
        config_yaml_path = model_path_map.get(modelVersion,model_path_map['yolov8'])  

    else: # for local
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
        train_path = data.get('train')
        val_path = data.get('val')
        desc = f"{desc}_" if desc else ''
        train_name = f"{desc}Train{len([f for f in os.listdir(train_path)if str(f).lower().endswith(('.jpg','.png','.jpeg'))])}_Val{len([f for f in os.listdir(val_path)if str(f).lower().endswith(('.jpg','.png','.jpeg'))])}"
        train_name = f"{time_stamp}-{train_name}"
        save_path = os.path.join(save_path, train_name)

    os.makedirs(save_path, exist_ok=True)
      
    if session_id:
        session_id = os.path.join(PJ_ROOT,"_api/logs/train",session_id)
        lgu.add(f'{session_id}.log', format=lgu_fmt, level="INFO")
    
    
    if rtd_yolo == 'yolo':
        model = YOLO(config_yaml_path,session_id = session_id).load(weight_path) if weight_path else YOLO(config_yaml_path,session_id = session_id)
    else:
        model = RTDETR(config_yaml_path).load(weight_path) if weight_path else RTDETR(config_yaml_path)
    results = model.train(
                data=data_yaml, 
                batch=batch_size or 8,
                device=device,
                project=save_path,
                epochs=epochs or 1, 
                imgsz=image_size or 640,
                cos_lr=True,
                lr0=learning_rate or 0.01,
                freeze=freeze or None,
                workers = 0,
                session_id = session_id
                # amp = False
    )
    if device != "cpu":
        torch.cuda.empty_cache()
    
    if web_params:
        for item in os.listdir(weights_dir:=os.path.join(save_path,'train/weights')):
            os.rename(os.path.join(weights_dir,item),os.path.join(weights_dir,f"{train_name}_{item}")) if str(item).endswith('.pt') else None

    
    lgu.info(f"训练结果保存到 {results.save_dir._str}")
    return results.save_dir._str

if __name__ == '__main__':
    standard_train(
                config_yaml_path='/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/12/yolo12.yaml',
                data_yaml='/home/panxiang/coding/kweilx/ultralytics/_api/data/dataset/yamls-202509161308-8fecdf0b-625a-4b3d-a721-c796a63399d4-original/mydata.yaml',
                batch_size=64,
                epochs = 1)