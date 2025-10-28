import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')

from ultralytics import YOLO
from ultralytics import RTDETR

import time
import torch
import gc
import yaml
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standard_train(config_yaml_path,
                   data_yaml='',
                   weight_path=None,
                   pr_name='',
                   desc='',
                   save_path=None,
                   batch_size=8,
                   epochs=500,
                   image_size=640,
                   learning_rate=0.01,
                   device=0,
                   freeze=None,
                   rtd_yolo='yolo',
                   stop_event=None) -> str:
    """
    改进的训练函数，支持停止控制
    """
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    train_path = data.get('train')
    val_path = data.get('val')
    
    desc = f"{desc}_" if desc else ''
    pr_name = f"Train{len([f for f in os.listdir(train_path) if str(f).lower().endswith(('.jpg','.png','.jpeg'))])}_Val{len([f for f in os.listdir(val_path) if str(f).lower().endswith(('.jpg','.png','.jpeg'))])}"
    PR_NAME = f"{desc}{pr_name}"
    TIME_STAMP = str(time.strftime("%Y%m%d%H%M", time.localtime()))
    BATCH_SIZE = batch_size or 8
    EPOCHS = epochs or 1
    IMAGE_SIZE = image_size or 640
    LEARNING_RATE = learning_rate or 0.01
    DEVICE = device or 0
    FREEZE = freeze or None
    
    # 检查停止信号
    if stop_event and stop_event.is_set():
        raise InterruptedError("训练在开始前被停止")
    
    if rtd_yolo == 'yolo':
        model = YOLO(config_yaml_path).load(weight_path) if weight_path else YOLO(config_yaml_path)
    else:
        model = RTDETR(config_yaml_path).load(weight_path) if weight_path else RTDETR(config_yaml_path)
    
    try:
        results = model.train(
            data=data_yaml, 
            batch=BATCH_SIZE,
            device='cpu',
            project=save_path,
            name=f"{TIME_STAMP}_{PR_NAME}",
            epochs=EPOCHS, 
            imgsz=IMAGE_SIZE,
            cos_lr=True,
            lr0=LEARNING_RATE,
            freeze=FREEZE
        )
        
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"训练结果保存到 {results.save_dir._str}")
        return results.save_dir._str
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        raise
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise