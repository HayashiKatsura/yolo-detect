import os
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import sys
sys.path.append(PJ_ROOT)

from ultralytics import YOLO
from ultralytics import RTDETR

import os
import torch
import gc
import copy
from tqdm import tqdm
import cv2
from uuid import uuid4
import time

from _project.mydata._code._utils.VideoDetectData import frame_detect_csv
from _api._utils.ImagestransferComponent.FromLocalImageFiles import TransferLocalImageFiles
from sqlmodel import Session,select
from _api.models.file import File as FileObject
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from _api.configuration.RedisConfig import redis_config

from datetime import datetime
import json

def standard_test(model_path:str,
                  test_image_path:str|dict,
                  save_folder:str = None,
                  conf:float=0.25,
                  batch_size:int=32,
                  record:bool=False,
                  only_one:bool = False,
                  save_detections:bool = False,
                  save_no_detections:bool = False,
                  rtd_yolo:str = 'yolo')->None:
    """
    标准测试流程，单模型，单图片 或 单文件夹
    Args:
    model_path: 模型路径
    test_image_path: 测试图片路径,可以是单张图片，也可以是文件夹
    save_folder: 保存结果文件夹，如果为None，则保存到模型文件夹下
    conf: 置信度阈值
    batch_size: 批处理大小
    record: 是否记录检测结果
    only_one: 是否只测试一个模型
    save_detections: 是否保存检测结果图片
    save_no_detections: 是否保存无检测结果图片
    """

    model_folder = None # 训练数据文件夹
    if str(model_path).endswith('.pt'):
        model_folder = os.path.dirname(os.path.dirname(model_path))
    else:
        model_folder = copy.deepcopy(model_path)
        model_path = os.path.join(model_folder,"weights","best.pt")
    save_folder = model_folder if save_folder is None else save_folder
    # save_folder = os.path.join(save_folder, f'predict_results_conf{conf}')
    model = YOLO(model_path) if rtd_yolo == 'yolo' else RTDETR(model_path)
    files_list = []
    if isinstance(test_image_path, str):
        if os.path.isfile(test_image_path) and str(test_image_path).lower().endswith(('.jpg', '.jpeg', '.png')):
            files_list.append(test_image_path)
        elif os.path.isdir(test_image_path):
            for item in os.listdir(test_image_path):
                if os.path.isfile(os.path.join(test_image_path,item)) and str(item).lower().endswith(('.jpg', '.jpeg', '.png')):
                    files_list.append(os.path.join(test_image_path,item))
        else:
            pass  
    elif isinstance(test_image_path, dict):
        files_list = test_image_path
    
    prediciton_results = []
    # 批量预测
    if isinstance(files_list, list): # for local
        for i in tqdm(iterable=range(0, len(files_list), batch_size), desc=f"测试文件{os.path.basename(test_image_path)} ，模型{os.path.basename(model_folder)}"):
            batch_files = files_list[i:i+batch_size]
            batch_paths = [f for f in batch_files]

            results = model.predict(batch_paths, conf=conf)
            os.makedirs(save_folder,exist_ok=True)
            for result in results:
                boxes = result.boxes 
                file_path = str(result.path)
                file_name = os.path.basename(file_path)

                if len(boxes)!= 0:
                    if record:
                        for _box in boxes:
                            _cls = int(_box.cls.tolist()[0])
                            _xywhn = _box.xywhn.tolist()[0]
                            with open(os.path.join(save_folder,f"{str(file_name).split('.')[0]}.txt"), mode='a') as f:
                                f.write(f'{_cls} {_xywhn[0]} {_xywhn[1]} {_xywhn[2]} {_xywhn[3]} {round(float(boxes[0].conf.tolist()[0]),2)}\n')
                    if save_detections:
                        result.save(os.path.join(save_folder, file_name))
                else:
                    if save_no_detections:
                        result.save(os.path.join(save_folder, file_name))
    
    elif isinstance(files_list, dict):  # for web    
        time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        random_id = str(uuid4().hex)
        dict_items = list(files_list.items())
        for i in tqdm(iterable=range(0, len(dict_items), batch_size), 
                    desc=f"测试文件 ,模型{os.path.basename(model_folder)}"):
            batch_items = dict_items[i:i+batch_size]
            batch_paths = [path for file_id, path in batch_items]
            
            # 创建路径到file_id的映射
            path_to_id = {path: file_id for file_id, path in batch_items}
            
            results = model.predict(batch_paths, conf=conf)
            save_folder = os.path.join(PJ_ROOT, '_api/data/predictions', f'{time_stamp}-{random_id}')
            os.makedirs(save_folder, exist_ok=True)
            
            for result in results:
                boxes = result.boxes 
                file_path = str(result.path)
                
                file_id = path_to_id[file_path]
                
                _name,_ext = os.path.splitext(os.path.basename(file_path))
                file_name = f'{_name}_detected{_ext}'
                txt_name = f'{_name}_detected.txt'

                if len(boxes) != 0:
                    height, width = boxes.orig_shape
                    prediciton_data = []
                    for _box in boxes:
                        _cls = int(_box.cls.tolist()[0])
                        _xywhn = _box.xywhn.tolist()[0]
                        _x = round((_xywhn[0]),2)
                        _y = round((_xywhn[1]),2)
                        _w = round((_xywhn[2]),2)
                        _h = round((_xywhn[3]),2)
                        
                        # YOLO 转为左上角坐标
                        x1 = int((float(_x) - float(_w) / 2) * width)
                        y1 = int((float(_y) - float(_h) / 2) * height)
                        x2 = int((float(_x) + float(_w) / 2) * width)
                        y2 = int((float(_y) + float(_h) / 2) * height)                        
                        
                        
                        prediciton_data.append(
                            {
                                'class':_cls,
                                'x': _x,
                                'y': _y,
                                'w': _w,
                                'h': _h,
                                'conf': round(float(_box.conf.tolist()[0]),2),
                                'predicted_area': abs(x1-x2)*abs(y1-y2),
                                'yolo_coord': [_x, _y, _w, _h],
                                'image_coord': [x1, y1, x2, y2],
                                'image_size': [height, width]
                            }
                        )
                        with open(os.path.join(save_folder, txt_name), mode='a') as f:
                            f.write(f'{_cls} {_xywhn[0]} {_xywhn[1]} {_xywhn[2]} {_xywhn[3]} {round(float(_box.conf.tolist()[0]), 2)}\n')
  
                    detected_image = os.path.join(save_folder, file_name)
                    result.save(detected_image)
                    prediciton_results.append(
                        {
                            'file_id': file_id,
                            'file_name': os.path.basename(file_path),
                            'count': len(result.boxes),
                            'prediction':prediciton_data,
                            # 'images':f"data:image/png;base64,{TransferLocalImageFiles(detected_image).toBase64()}"
                            'images':detected_image,
                            'labels':os.path.join(save_folder, txt_name),
                            'timestamp':str(time.strftime("%Y%m%d%H%M", time.localtime()))
                        }
                    )   
                    
                else:
                     prediciton_results.append(
                            {
                                'file_id': file_id,
                                'file_name': os.path.basename(file_path),
                                'count': 0,
                                'timestamp':str(time.strftime("%Y%m%d%H%M", time.localtime()))
                            }
                        )
        
        # 写入数据库
        with Session(SQL_ENGINE) as session:
            for result in prediciton_results:
                query = select(FileObject).where(FileObject.id == result['file_id'])
                file_data = session.exec(query).first()

                if file_data:
                    if not file_data.media_annotations:
                        file_data.media_annotations = [result]
                    else:
                        file_data.media_annotations = [result] + file_data.media_annotations 
                    file_data.updated_at = datetime.utcnow()
            session.commit()
        
        # 缓存结果
        redis_client = redis_config.get_client()
        redis_key = f"prediction:{datetime.now().date()}"
        cached_data = redis_client.get(redis_key)
        if cached_data:
            cached_data = json.loads(cached_data)
            for result in prediciton_results:
                file_id = result['file_id']
                existing_entry = next((entry for entry in cached_data if entry['file_id'] == file_id), None)
                existing_entry.update(result) if existing_entry else cached_data.append(result)
        else:
            cached_data = prediciton_results
        # redis_client.setex(redis_key,3600*24, json.dumps(cached_data))
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match=f"{redis_key}*")
            for key in keys:
                redis_client.setex(key,3600*24, json.dumps(cached_data))
            if cursor == 0: 
                break
                    
    if only_one:
        torch.cuda.empty_cache()      
        gc.collect()   

    return prediciton_results
def standard_test_video(model_path:str,
                  test_image_path:str,
                  save_folder:str = None,
                #   conf:float=0.25,
                #   batch_size:int=32,
                #   record:bool=False,
                  only_one:bool = True,
                #   save_detections:bool = False,
                #   save_no_detections:bool = False,
                  rtd_yolo:str = 'yolo')->None:
    """
    标准测试流程，单模型，单图片 或 单文件夹
    Args:
    model_path: 模型路径
    test_image_path: 测试图片路径,可以是单张图片，也可以是文件夹
    save_folder: 保存结果文件夹，如果为None，则保存到模型文件夹下
    conf: 置信度阈值
    batch_size: 批处理大小
    record: 是否记录检测结果
    only_one: 是否只测试一个模型
    save_detections: 是否保存检测结果图片
    save_no_detections: 是否保存无检测结果图片
    
    return: 保存结果文件夹路径, csv 文件路径
    """
    

    model_folder = None # 训练数据文件夹
    if str(model_path).endswith('.pt'):
        model_folder = os.path.dirname(os.path.dirname(model_path))
    else:
        model_folder = copy.deepcopy(model_path)
        model_path = os.path.join(model_folder,"weights","best.pt")
    save_folder = model_folder if save_folder is None else save_folder
    # save_folder = os.path.join(save_folder, f'predict_results_conf{conf}')
    model = YOLO(model_path) if rtd_yolo == 'yolo' else RTDETR(model_path)
    files_list = []
    if os.path.isfile(test_image_path) and str(test_image_path).lower().endswith(('.mp4', '.avi', '.mov')):
        files_list.append(test_image_path)
    elif os.path.isdir(test_image_path):
        for item in os.listdir(test_image_path):
            if os.path.isfile(os.path.join(test_image_path,item)) and str(item).lower().endswith(('.mp4', '.avi', '.mov')):
                files_list.append(os.path.join(test_image_path,item))
    else:
        pass  
    
    """
    新增内容
    """
    for video_path in files_list:
        target_frames = []  # 记录含目标(置信度≥0.3）的帧
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    

        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        
        # 初始化视频写入器(H.264编码,分辨率偶数处理）
        fourcc = cv2.VideoWriter_fourcc(*"H264") # TODO 编码器问题，待处理
        detect_video_path = os.path.join(save_folder,f"{os.path.basename(video_path).split('.')[0]}_detected.mp4")
        out = cv2.VideoWriter(detect_video_path, fourcc, fps, (width, height), isColor=True)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 调整帧大小(与输出视频一致）
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # 模型检测(置信度≥0.3,过滤低置信目标）
            results = model(frame, conf=0.3)
            annotated_frame = results[0].plot()
            
            # 记录含目标(置信度≥0.3）的帧信息
            frame_detections = []
            for box in results[0].boxes:
                det_conf = float(box.conf[0])
                if det_conf >= 0.3:  # 仅保留置信度≥0.3的目标
                    frame_detections.append({
                        "class": model.names[int(box.cls[0])],
                        "confidence": det_conf
                    })
            if frame_detections:  # 仅当帧含有效目标时记录
                target_frames.append({
                    "frame_idx": frame_count,
                    "timestamp_ms": int((frame_count / fps) * 1000),
                    "detections": frame_detections
                })

            # 写入标注视频 + 更新处理进度
            out.write(annotated_frame)
            frame_count += 1
            # processing_progress[task_id] = min(int((frame_count / total_frames) * 100), 100)
        
        # 视频处理完成后,生成CSV(确保仅执行一次）
        if target_frames:
            csv_path = frame_detect_csv(save_folder, target_frames)
        else:
            # print(f"任务 {task_id}：未检测到置信度≥0.3的目标,不生成CSV")
            csv_path = None
        
        # 释放资源(避免文件占用）
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # 接口部分(不变,确保前端能正常上传、查进度、取结果）
     
    if only_one:
        torch.cuda.empty_cache()      
        gc.collect()   

    return detect_video_path,csv_path

def batch_test(
                test_data_list:list,
                test_image_path=None,
                conf:float=0.25,
                batch_size:int=32,
                save_detections:bool = False,
                save_no_detections:bool = False,
                rtd_yolo:str = 'yolo'
                )->None:
    """
    标准测试流程，多模型，多图片 或 多文件夹
    Args:   
    test_data_list,元素为dict，参数符合standard_test函数
    """
    
    for test_data in test_data_list:
        model_path = test_data['model_path']
        test_image_path = test_data.get('test_image_path',test_image_path)
        save_folder = test_data.get('save_folder',None)
        conf = test_data.get('conf',conf)
        record = test_data.get('record',True)
        batch_size = test_data.get('batch_size',batch_size)
        rtd_yolo = test_data.get('rtd_yolo',rtd_yolo)
        save_no_detections = test_data.get('save_no_detections',save_no_detections)
        standard_test(  
                    model_path = model_path,
                    test_image_path = test_image_path,
                    save_folder = save_folder,
                    conf = conf,
                    record = record,
                    batch_size = batch_size,
                    only_one = False,
                    rtd_yolo = rtd_yolo
                        )
        
    torch.cuda.empty_cache()      
    gc.collect()   
    
if __name__ == '__main__':
    import time
    time_cost = time.time()
    standard_test_video(model_path = '/home/panxiang/coding/kweilx/ultralytics/zwx.pt',
                  test_image_path='/home/panxiang/coding/kweilx/ultralytics/1/1.mp4',
                  save_folder = '/home/panxiang/coding/kweilx/ultralytics/2',
                #   conf=0.5,
                #   batch_size=200,
                #   record=True,
                  only_one = False,
                #   save_detections=True,
                #   save_no_detections=False,
                  rtd_yolo= 'yolo')
    time_cost = time.time() - time_cost
    print(f"测试用时:{time_cost:.2f}秒")