import os
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import sys
sys.path.append(project_path)

from ultralytics import YOLO
from ultralytics import RTDETR

import os
import torch
import gc
import copy
from tqdm import tqdm
import cv2

from _project.mydata._code._utils.VideoDetectData import frame_detect_csv

def standard_test(model_path:str,
                  test_image_path:str,
                  save_folder:str = None,
                  conf:float=0.25,
                  batch_size:int=32,
                  record:bool=False,
                  only_one:bool = True,
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
    if os.path.isfile(test_image_path) and str(test_image_path).lower().endswith(('.jpg', '.jpeg', '.png')):
        files_list.append(test_image_path)
    elif os.path.isdir(test_image_path):
        for item in os.listdir(test_image_path):
            if os.path.isfile(os.path.join(test_image_path,item)) and str(item).lower().endswith(('.jpg', '.jpeg', '.png')):
                files_list.append(os.path.join(test_image_path,item))
    else:
        pass  
    
    # 批量预测
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


    if only_one:
        torch.cuda.empty_cache()      
        gc.collect()   

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