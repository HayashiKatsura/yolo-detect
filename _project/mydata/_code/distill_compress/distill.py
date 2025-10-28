import os
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')

import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

import time
TIME_STAMP = str(time.strftime("%Y%m%d%H%M", time.localtime()))



if __name__ == '__main__':
    DESC = 'convnextv2-mutilscaleedgeinfomation-prune-BCKD-2.0'
    param_dict = {
            # origin
            'model': '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/202508251810-prune-slim-convnextv2-mutilscale36912-reg0.05-gt-prune/weights/prune.pt',
            'data':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_datasets/anomolies/new_source_total/mydata.yaml',
            'imgsz': 640,
            'epochs': 200,
            'batch': 32,
            'workers': 1,
            'cache': False,
            'optimizer': 'SGD',
            'device': '0',
            'close_mosaic': 20,
            # 'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
            'project':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/disstillation',
            'name':f'{TIME_STAMP}-{DESC}',
            
            # distill
            'prune_model':True,
            'teacher_weights': '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508052210_yolo12-convnextv2-mutilscaleedgeinfomation_Train2419_Val479/weights/best.pt',
            'teacher_cfg': '/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/models/chips/yolo12-convnextv2-mutilscaleedgeinfomation.yaml',
            # 'kd_loss_type': 'logical',
            'kd_loss_type': 'all',
            'kd_loss_decay': 'constant',
            
            'logical_loss_type': 'BCKD',
            'logical_loss_ratio': 2.0,
            
            'teacher_kd_layers': '3,6,9,12',
            'student_kd_layers': '3,6,9,12',
            'feature_loss_type': 'cwd',
            'feature_loss_ratio': 1.0
        }
        
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()