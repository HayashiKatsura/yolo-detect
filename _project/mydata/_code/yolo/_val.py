import os
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import sys
sys.path.append(PJ_ROOT)

from ultralytics import YOLO,RTDETR
import os
import shutil
import torch
import gc
import glob
from _api._utils.ImagestransferComponent.FromLocalImageFiles import TransferLocalImageFiles
from concurrent.futures import ThreadPoolExecutor
from sqlmodel import Session
from datetime import datetime
from sqlalchemy import Column, JSON
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from sqlmodel import select
from _api.models.file import File as FileObject
import time

def standard_val(val_data:dict,only_one = True,save_folder=None)->None:
    """
    标准验证流程
    传入字典参数
    必须参数：
    1、model：训练文件的根目录路径
    2、yaml_data：数据配置文件路径
    可选参数：
    其中，CONF_LIST为验证的置信度阈值列表，默认为[0.25]
    """
    BATCH_SIZE = val_data.get('batch_size',8)
    IMAGE_SIZE = val_data.get('image_size',640)
    DEVICE = val_data.get('device','cuda:0' if torch.cuda.is_available() else 'cpu')
    PLOTS = val_data.get('plots',True)
    MODEL = val_data['model']
    MODEL = os.path.join(MODEL,'weights','best.pt') if not str(MODEL).endswith('.pt') else MODEL
    YAML_DATA = val_data['yaml_data']
    WEIGHT_ID = val_data.get('weight_id',None)
    # YAML_DATA = glob.glob(os.path.join(YAML_DATA, '*.yaml'))[0]
    
    DESC = val_data.get('desc','new_val')
    CONF_LIST = val_data.get('conf_list',[0.25])
    RTD_YOLO= val_data.get('rtd_yolo','yolo')
    if RTD_YOLO == 'yolo':
        model = YOLO(MODEL) 
    else:
        model = RTDETR(MODEL) 
    results = []
    val_images = []
    try:
        for _CONF in CONF_LIST:
            SAVE_FOLDER = os.path.join(val_data['model'],f'{DESC}',f'{_CONF}_VAL') if not save_folder else save_folder
            if os.path.exists(SAVE_FOLDER):
                shutil.rmtree(SAVE_FOLDER)
            os.makedirs(SAVE_FOLDER,exist_ok=True)
            
            # 核心验证流程
            metrics = model.val(
                        data=YAML_DATA,
                        imgsz=IMAGE_SIZE,
                        batch=BATCH_SIZE,
                        project=SAVE_FOLDER,
                        device=DEVICE,
                        plots=PLOTS,
                        conf = _CONF)

            #  写到本地
            ap50_list = []
            with open(os.path.join(SAVE_FOLDER,'metrics.txt'), mode='a') as f:
                f.write(f'conf: {_CONF}\n')
                f.write(f'mAP50: {round(float(metrics.box.map50),2)}\n')
                f.write(f'mAP75: {round(float(metrics.box.map75),2)}\n')
                f.write(f'mAP50-95: {round(float(metrics.box.map),2)}\n')
                f.write(f'mAP50: {round(float(metrics.results_dict["metrics/mAP50(B)"]),2)}\n')
                f.write(f'mAP50-95: {round(float(metrics.results_dict["metrics/mAP50-95(B)"]),2)}\n')
                for index, item in enumerate(metrics.box.ap50.tolist()):
                    f.write(f'{metrics.names[index]}: {round(float(item),2)}\n')
                    ap50_list.append({'class': metrics.names[index], 'ap50': round(float(item),2)})
                f.write(f'mean-ap50: {round(float(sum(metrics.box.ap50.tolist())/len(metrics.box.ap50.tolist())),2)}\n')
                f.write(f'precision: {round(float(metrics.results_dict["metrics/precision(B)"]),2)}\n')
                f.write(f'recall: {round(float(metrics.results_dict["metrics/recall(B)"]),2)}\n')
                f.flush()  # 强制刷新缓冲区
            
            # 获取图像
            val_images_list = [
                        'BoxF1_curve.png',
                        'BoxP_curve.png',
                        'BoxR_curve.png',
                        'BoxPR_curve.png',
                        'confusion_matrix_normalized.png',
                        'confusion_matrix.png',
                        'F1_curve.png',
                        'P_curve.png',
                        'R_curve.png',
                        'PR_curve.png',
                        'val_batch0_labels.jpg',
                        'val_batch0_pred.jpg',
                        'val_batch1_labels.jpg',
                        'val_batch1_pred.jpg',
                        'val_batch2_labels.jpg',
                        'val_batch2_pred.jpg',
            ]
            def load_image_if_exists(item):
                img_path = os.path.join(SAVE_FOLDER, 'val', item)
                if os.path.exists(img_path):
                    try:
                        return {
                            'name': item,
                            'image': f"data:image/png;base64,{TransferLocalImageFiles(img_path).toBase64()}"
                        }
                    except Exception as e:
                        print(f"加载 {item} 失败: {e}")
                return None
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                val_images = [img for img in executor.map(load_image_if_exists, val_images_list) if img]
             
            # 返回数据
            results.append(
                {
                    'conf': _CONF,
                    'mAP50': round(float(metrics.box.map50),2),
                    'mAP75': round(float(metrics.box.map75),2),
                    'mAP50-95': round(float(metrics.box.map),2),
                    'ap50': ap50_list,
                    'class_ap50': ap50_list,
                    'mean-ap50': round(float(sum(metrics.box.ap50.tolist())/len(metrics.box.ap50.tolist())),2),
                    'precision': round(float(metrics.results_dict["metrics/precision(B)"]),2),
                    'recall': round(float(metrics.results_dict["metrics/recall(B)"]),2),
                    'save_folder': SAVE_FOLDER,
                    'weight_id': WEIGHT_ID,
                    'val_images': val_images,
                    'timestamp': str(time.strftime("%Y%m%d%H%M", time.localtime()))
                }
            )
            
            # 写入数据库
            if WEIGHT_ID:
                with Session(SQL_ENGINE) as session:
                    query = select(FileObject.id, FileObject.model_metrics).where(FileObject.id == WEIGHT_ID)
                    weight_data = session.exec(query).first()

                    if not weight_data:
                        raise ValueError(f"权重文件ID {WEIGHT_ID} 不存在")  # 使用更具体的异常类型

                    if not weight_data.model_metrics:
                        weight_data.model_metrics = results
                    else:
                        weight_data.model_metrics.append(results[0])

                    weight_data.updated_at = datetime.utcnow()

                    session.commit()  # 这是关键，提交修改

             
        if only_one: 
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        pass
    
    # return SAVE_FOLDER
    return results
    
    
def batch_val(results_path:dict|list)->None:
    """
    批量验证，传入一个列表，元素为字典，符合标准验证流程的参数
    Args:
        results_path (dict | list): _description_
    """
    results_path = [results_path] if isinstance(results_path, dict) else results_path
    for _results_path in results_path:
        standard_val(_results_path,only_one=False)

  
            
if __name__ == '__main__':
    results_path = [
        {
        'model':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_results/prune/202508271900-prune-lamp-convnextv2-mutilscale36912-reg0.05-gt-sp4.0-finetune',
        'yaml_data':'/home/panxiang/coding/kweilx/ultralytics/_project/mydata/_datasets/anomolies/new_source_total/mydata.yaml'
        },
    ]
                         
    batch_val(results_path)