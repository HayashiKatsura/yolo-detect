import os
import sys
sys.path.append('/home/panxiang/coding/kweilx/ultralytics')

import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
# from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
# from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
# from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune
import time


def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    TIME_STAMP = str(time.strftime("%Y%m%d%H%M", time.localtime()))
    
    sp_method = 'slim'
    DESC = f'{sp_method}-convnextv2-mutilscale36912-reg0.01-gt-sp2.0'
    param_dict = {
        # # origin
        # # ok
        # 'model': '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060012_yolo12_Train2419_Val479/weights/best.pt',
        # # ok
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508031302_yolo12-Newconvnextv2-GatebBlockx0_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508071600_yolo12-convnextv2-mutilscaleedgeinfomation3579_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508060215_yolo12-Newconvnextv2-mutilscaleedgeinfomation_Train2419_Val479/weights/best.pt',
       'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508052210_yolo12-convnextv2-mutilscaleedgeinfomation_Train2419_Val479/weights/best.pt',
        # erro
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508021941_yolo12-Newconvnextv2-GatebBlockx4_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508032110_yolo12-Newconvnextv2-GatebBlockx1_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060224_yolo12-convnextv2_Train2419_Val479/weights/best.pt',
        'data':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_datasets/anomolies/new_source_total/mydata.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 16,
        # 'batch': 8,
        'workers': 1,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        # 'device': 'cpu',
        'close_mosaic': 0,
        'project':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/slim',
        'name':f'{TIME_STAMP}-prune-{DESC}',
        # 'amp':False,
        
        # prune group_sl
        # 'prune_method':'lamp', 
        'prune_method':f'{sp_method}',
        'global_pruning': True,
        'speed_up': 2.0,
        # 'reg': 0.0005,
        'reg': 0.01,
        'reg_decay':0.05,
        'sl_epochs': 500,
        'sl_hyp': '/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)
    
    TIME_STAMP = str(time.strftime("%Y%m%d%H%M", time.localtime()))
    
    sp_method = 'slim'
    DESC = f'{sp_method}-convnextv2-mutilscale36912-reg0.05-gt-sp2.0'
    param_dict = {
        # # origin
        # # ok
        # 'model': '/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060012_yolo12_Train2419_Val479/weights/best.pt',
        # # ok
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508031302_yolo12-Newconvnextv2-GatebBlockx0_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508071600_yolo12-convnextv2-mutilscaleedgeinfomation3579_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508060215_yolo12-Newconvnextv2-mutilscaleedgeinfomation_Train2419_Val479/weights/best.pt',
       'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508052210_yolo12-convnextv2-mutilscaleedgeinfomation_Train2419_Val479/weights/best.pt',
        # erro
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508021941_yolo12-Newconvnextv2-GatebBlockx4_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202508032110_yolo12-Newconvnextv2-GatebBlockx1_Train2419_Val479/weights/best.pt',
        # 'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/yolo12n/multi_classes/202507060224_yolo12-convnextv2_Train2419_Val479/weights/best.pt',
        'data':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_datasets/anomolies/new_source_total/mydata.yaml',
        'imgsz': 640,
        'epochs': 300,
        'batch': 16,
        # 'batch': 8,
        'workers': 1,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        # 'device': 'cpu',
        'close_mosaic': 0,
        'project':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/slim',
        'name':f'{TIME_STAMP}-prune-{DESC}',
        # 'amp':False,
        
        # prune group_sl
        # 'prune_method':'lamp', 
        'prune_method':f'{sp_method}',
        'global_pruning': True,
        'speed_up': 2.0,
        # 'reg': 0.0005,
        'reg': 0.05,
        'reg_decay':0.05,
        'sl_epochs': 500,
        'sl_hyp': '/home/panxiang/coding/kweilx/ultralytics/ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)
    
    
