import os
# 获取当前项目所在路径
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import sys
sys.path.append(project_path)

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
from _project.mydata._code.yolo._val import standard_val
from _project.mydata._code._utils.files_process.videos_process import get_videos_frame_count

from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error


parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class YoloAPI:
    def __init__(self,weight_id=None,conf=0.25,camera=False):
        self.log_save_path = os.path.join(parent_folder, 'data', 'logs')
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        self.weight_id = weight_id
        self.camera = camera
        if self.camera and self.camera!='null':
            self.camera = True
        if self.weight_id:
            # self.weight_path = files_info("file_id", 
            #                               self.weight_id,
            #                               record_path=files_record_mapping()[str(FilesType.weights)][1]).get('file_path',None)
            self.weight_path = WeightTable.query.filter_by(file_id=self.weight_id).first().file_path
        else:
            self.weight_path = None
        self.conf = conf
        self.temp_folder = os.path.join(parent_folder, 'data', 'temp') # 存放临时文件
        os.makedirs(self.temp_folder, exist_ok=True)
      
        
    @handle_db_error
    def detect_file(self,image_id,batch_size=32,save_folder=None,record=True):
        
        # 临时文件夹
        temp_folder = os.path.join(self.temp_folder, f"temp-{self.time_stamp}-{str(uuid4())[20:]}")
        os.makedirs(temp_folder, exist_ok=True)
        
        if str(image_id).find(',')!=-1: # ->多id
            file_id_path_map = {}
            
            image_id_list = image_id.split(',')
            for item_id in image_id_list:
                # current_file_path = files_info("file_id",
                #                           str(item_id),
                #                           record_path=files_record_mapping()[str(FilesType.cameras)][1])['file_path']
                current_file_path = FileTable.query.filter_by(file_id=str(item_id)).first().file_path
                
                
                if os.path.isfile(current_file_path):
                    file_name = os.path.basename(current_file_path)
                    temp_file_path = os.path.join(temp_folder, file_name)
                    shutil.copy(current_file_path, temp_file_path)
                    file_id_path_map.update({str(file_name):str(item_id)})
                    
            image_path = temp_folder
        
        else: # ->单id
            file_id_path_map = {}
            
            if str(image_id).find('folder')!=-1: # ->文件夹
                # image_path = files_info("file_folder_id",
                #                         image_id,
                #                         record_path=files_record_mapping()[str(FilesType.images)][1]).get('file_folder_path',None)
                
                # folder_info = FileTable.query.with_entities(FileTable.folder_path).all()
                folder_info = FileTable.query.with_entities(FileTable.file_id,FileTable.file_path,FileTable.folder_path).all()
                for item in folder_info:
                    _file_id, _file_path, _folder_path = item
                    if str(_folder_path).find(image_id)!=-1:
                        image_path = _folder_path
                        file_id_path_map.update(
                            {str(os.path.basename(_file_path)):str(_file_id)}
                        )
        
                
            else: # ->文件 
                # image_path = files_info("file_id", 
                #                         image_id,
                #                         record_path=files_record_mapping()[str(FilesType.images)][1]).get('file_path',None)
                image_path = FileTable.query.filter_by(file_id=image_id).first().file_path
                
        
        save_folder = os.path.join(parent_folder, 'data', 'predict',f"predict-{self.time_stamp}-{str(uuid4())[20:]}")
        os.makedirs(save_folder, exist_ok=True)
        
        # 视频类型的文件
        if os.path.isfile(image_path):
            if os.path.splitext(image_path)[1] in ['.mp4','.avi','.mkv']:
                detect_video_path,csv_path = standard_test_video(
                    model_path = self.weight_path,
                    test_image_path = image_path,
                    save_folder = save_folder,
                )
                
                # 检测结果写入数据库
                detected_id= f'detect-{self.time_stamp}-{str(uuid4())}' # 检测结果id
                _video = FileTable.query.filter_by(file_id=image_id).first()
                if _video:
                    _video.is_detected = str(detected_id)
                
                from datetime import datetime,timezone
                _detections = DetectionTable(
                            file_id = image_id,
                            weight_id = self.weight_id,
                            details = {
                                "detect_image_path":detect_video_path,
                                "detect_txt_path":csv_path
                                }
                            )
                db.session.add(_detections)
                db.session.commit()
                return {"detect_id":detected_id}
            
        elif os.path.isdir(image_path):# TODO
            pass
        
        
        # 一般意义上的图像文件
        standard_test(
            model_path=self.weight_path,
            test_image_path=image_path,
            conf=self.conf,
            batch_size=batch_size,
            save_folder=save_folder,
            save_detections=True,
            record=True,
        )
        success = False
        results = []
        
        if os.path.isdir(image_path):
            image_files = glob.glob(f'{save_folder}/*.jpg') + glob.glob(f'{save_folder}/*.png') + glob.glob(f'{save_folder}/*.jpeg')
            for image_file in image_files:
                success = False
                try:
                    detect_image_base64 = ''
                    height, width = cv2.imread(image_file).shape[:2]
                    detect_image_path = image_file
                    # detect_image_id= f'image-{self.time_stamp}-{str(uuid4())}' # 检测输出的图像id
   
                    detect_image_base64 =  f'data:image/png;base64,{TransferLocalImageFiles(detect_image_path).toBase64()}',
                    detect_image_base64 = detect_image_base64[0] if isinstance(detect_image_base64,tuple) else detect_image_base64
                    
                    detect_txt_path = f'{os.path.splitext(detect_image_path)[0]}.txt'
                    # detect_txt_id= f'txt-{self.time_stamp}-{str(uuid4())}'
                    
                    # files_info("file_id", 
                    #         file_id_path_map[str(os.path.basename(image_file))], 
                    #         "is_detected", 
                    #         f'{detect_image_id},{detect_txt_id}',
                    #         record_path=files_record_mapping()[str(FilesType.cameras) if self.camera else str(FilesType.images)][1])
                    
                    # is_detected_map = {detect_image_id:detect_image_path,detect_txt_id:detect_txt_path}
                    # record_msg = \
                    #         {
                    #             'file_id': file_id_path_map[str(os.path.basename(image_file))],
                    #             'weight_id':self.weight_id,
                    #             'is_detected':is_detected_map,
                    #             'file_create_time':str(self.time_stamp)
                    #         }
                    # data_record(record_msg,fieldnames=table_2(),save_path=files_record_mapping()[str(FilesType.detect)][1])
                    
                    detected_id= f'detect-{self.time_stamp}-{str(uuid4())}' # 检测结果id
            
                    _image = FileTable.query.filter_by(file_id=file_id_path_map[str(os.path.basename(image_file))]).first()
                    if _image:
                        _image.is_detected = str(detected_id)
                        
                    _detections = DetectionTable(
                                file_id = str(file_id_path_map[str(os.path.basename(image_file))]),
                                weight_id = self.weight_id,
                                details = {"detect_image_path":detect_image_path,"detect_txt_path":detect_txt_path})
                    db.session.add(_detections)
                    db.session.commit()
                            
                    
                    with open(detect_txt_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            _cls,_x,_y,_w,_h,_conf = line.strip().split(' ')
                            
                            # YOLO 转为左上角坐标
                            x1 = int((float(_x) - float(_w) / 2) * width)
                            y1 = int((float(_y) - float(_h) / 2) * height)
                            x2 = int((float(_x) + float(_w) / 2) * width)
                            y2 = int((float(_y) + float(_h) / 2) * height)
                            # 计算面积
                            detect_area = abs(x1-x2)*abs(y1-y2)
                            results.append(
                                {
                                    'file_name':str(os.path.basename(detect_image_path)),
                                    'cls':_cls,
                                    'conf':_conf,
                                    'yolo_coord':f'(x:{round(float(_x),2)},y:{round(float(_y),2)},w:{round(float(_w),2)},h:{round(float(_h),2)})',
                                    'detect_coord':f'(x1:{x1},y1:{y1},x2:{x2},y2:{y2})',
                                    'detect_area':detect_area,
                                    'image_size':f'height:{height},width:{width}',
                                    'detect_image_base64':detect_image_base64,
                                    'file_path':str(os.path.dirname(detect_image_path))
                                }
                            )
                        
                    success = True
                except Exception as e:
                    pass     
                # if success:
                #     files_info("file_path", 
                #                os.path.join(image_path, 
                #                             os.path.basename(image_file)), 
                #                "is_detected", 
                #                f'{detect_image_id},{detect_txt_id}',
                #                record_path=files_record_mapping()[str(FilesType.cameras) if self.camera else str(FilesType.images)][1])
            
            
            return results[0]     
          
        elif os.path.isfile(image_path):
            # 当为单个文件时
            detect_image_base64 = ''
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            results = []
            try:
                image_files = glob.glob(f'{save_folder}/*.jpg') + glob.glob(f'{save_folder}/*.png') + glob.glob(f'{save_folder}/*.jpeg')
                detect_image_path = image_files[0]
                # detect_image_id= f'image-{self.time_stamp}-{str(uuid4())}'
                
                detect_image_base64 =  f'data:image/png;base64,{TransferLocalImageFiles(detect_image_path).toBase64()}',
                detect_image_base64 = detect_image_base64[0] if isinstance(detect_image_base64,tuple) else detect_image_base64
                
                
                detect_txt_path = f'{os.path.splitext(detect_image_path)[0]}.txt'
                # detect_txt_id= f'txt-{self.time_stamp}-{str(uuid4())}'                
                
                with open(detect_txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        _cls,_x,_y,_w,_h,_conf = line.strip().split(' ')
                        
                        # YOLO 转为左上角坐标
                        x1 = int((float(_x) - float(_w) / 2) * width)
                        y1 = int((float(_y) - float(_h) / 2) * height)
                        x2 = int((float(_x) + float(_w) / 2) * width)
                        y2 = int((float(_y) + float(_h) / 2) * height)
                        # 计算面积
                        detect_area = abs(x1-x2)*abs(y1-y2)
                        results.append(
                            {
                                'file_name':str(os.path.basename(detect_image_path)),
                                'cls':_cls,
                                'conf':_conf,
                                'yolo_coord':f'(x:{round(float(_x),2)},y:{round(float(_y),2)},w:{round(float(_w),2)},h:{round(float(_h),2)})',
                                'detect_coord':f'(x1:{x1},y1:{y1},x2:{x2},y2:{y2})',
                                'detect_area':detect_area,
                                'image_size':f'height:{height},width:{width}',
                                'detect_image_base64':detect_image_base64,
                                'file_path':str(os.path.dirname(detect_image_path))
                            }
                        )
                success = True
            except Exception as e:
                pass
                
            if success:
                
                # files_info("file_id", 
                #            image_id, 
                #            "is_detected", 
                #            f'{detect_image_id},{detect_txt_id}',
                #            record_path=files_record_mapping()[str(FilesType.images)][1])
                # is_detected_map = {detect_image_id:detect_image_path,detect_txt_id:detect_txt_path}
                # record_msg = \
                #         {
                #             'file_id': image_id,
                #             'weight_id':self.weight_id,
                #             'is_detected':is_detected_map,
                #             'file_create_time':str(self.time_stamp)
                #         }
                # data_record(record_msg,fieldnames=table_2(),save_path=files_record_mapping()[str(FilesType.detect)][1])
                
                detected_id= f'detect-{self.time_stamp}-{str(uuid4())}' # 检测结果id
                _image = FileTable.query.filter_by(file_id=image_id).first()
                if _image:
                    _image.is_detected = str(detected_id)
                
                from datetime import datetime,timezone
                _detections = DetectionTable(
                            file_id = image_id,
                            weight_id = self.weight_id,
                            details = {
                                "detect_image_path":detect_image_path,
                                "detect_txt_path":detect_txt_path
                                }
                            )

                
                db.session.add(_detections)
                db.session.commit()
                return results
    
    def detect_progress(self,file_id):
        original_file = FileTable.query.filter_by(file_id=str(file_id)).first().file_path
        detected_file = DetectionTable.query.filter_by(file_id=str(file_id)).first().details.get('detect_image_path',None)
        if original_file and detected_file:
            return int(get_videos_frame_count(detected_file)/get_videos_frame_count(original_file))
        return 0
        
    
    def val_weight(self,dataYamlId):
        # self.data_yaml_path = files_info("file_id", 
        #                                  dataYamlId,
        #                                  record_path=files_record_mapping()[str(FilesType.yamls)][1]).get('file_path',None)
        
        self.data_yaml_path = DatasetTable.query.filter_by(file_id=dataYamlId).first().file_path
        # 验证数据存储文件夹
        save_folder = os.path.join(parent_folder, 'data', 'val',f"val-folder-{self.conf}-{self.time_stamp}-{str(uuid4())[20:]}")
        os.makedirs(save_folder, exist_ok=True)
        results = standard_val({
                                        'model':self.weight_path,
                                        'yaml_data':self.data_yaml_path,
                                        'conf_list':[self.conf]
                                        },
                                save_folder=save_folder)  
        if results:
            # 确认权重
            # files_info("file_id", 
            #            self.weight_id, 
            #            "is_detected", 
            #            str(results),
            #            record_path=files_record_mapping()[str(FilesType.weights)][1]) 
            _weight = WeightTable.query.filter_by(file_id=self.weight_id).first()
            if _weight:
                _weight.is_validated = str(results)
                db.session.commit()
            
            # metrics
            metrics_reults = {}
            with open(os.path.join(save_folder,'metrics.txt'), 'r') as f:
                metrics = f.readlines()
                for metric in metrics:
                    metric_name, metric_value = metric.strip().split(': ')
                    metrics_reults.update({metric_name:metric_value})
            
            # val_image
            file_name_list = [
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

            val_images = [
                f"data:image/png;base64,{TransferLocalImageFiles(os.path.join(save_folder,'val',file_name)).toBase64()}" 
                for file_name in file_name_list if os.path.exists(_image:=os.path.join(save_folder,'val',file_name)) and os.path.isfile(_image) 
            ]
            
            return {'metrics':metrics_reults,'val_images':val_images}   
            
        return {}
        
        
    def validate(self,file_id):
        """
        验证视频 临时函数
        """
        
        _video_detected = DetectionTable.query.filter_by(file_id=file_id).first().details
        _detect_txt_path = _video_detected.get('detect_txt_path',None)
        detect_table = []
        if _detect_txt_path:
            with open(_detect_txt_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    detect_table.append(dict(row))
            
        
            return {"file_id":file_id,"detect_table":detect_table}
        
                
        
    
    def train_log(self,args):
        log_file_id = args['log_file_id']
        line_number = int(args['line_num'])
        log_file_id = FilesID('log_file_id',log_file_id)
        log_file = is_files_exist(log_file_id) # 一个
        log_file_path = log_file[0]['data'][0]['file_path']
        if not log_file_path:
            return {}
        log_file_path = os.path.join(log_file_path, 'train','results.csv')
        return get_line_data(log_file_path,line_number)
        
    def train(self,args):
        # TODO 存在几个无默认值
        # time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        YAML_DATA = args['yaml_path']
        SAVE_FOLDER = args['save_folder']
        
        # PR_NAME = args.get('pr_name', None) or '_Train'
        
        # TIME_STAMP = time.strftime("%Y%m%d%H%M", time.localtime())
        BATCH_SIZE = int(args.get('batch_size', 8))
        EPOCHS = int(args.get('epochs', 500))
        IMAGE_SIZE = int(args.get('image_size', 640))    
        LEARING_RATE = float(args.get('learning_rate', 0.01))
        # DEVICE = int(args.get('device', 0))
        DEVICE = args.get('device', None) or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        yolo_model = YOLO(os.path.join(project_path, 'ultralytics/cfg/models/v8/yolov8.yaml')).load(os.path.join(project_path, 'yolov8n.pt'))  # build from YAML and trainsfer weights
        # Train the model
        results = yolo_model.train(data=YAML_DATA, 
                            batch=BATCH_SIZE,
                            device=DEVICE,
                            project=SAVE_FOLDER,
                            # name=f"{PR_NAME}_{TIME_STAMP}",
                            epochs=EPOCHS, 
                            imgsz=IMAGE_SIZE,
                            cos_lr=True,
                            lr0=LEARING_RATE)
        del yolo_model
        torch.cuda.empty_cache()
        
        # 训练文件夹
        train_folder_record_msg = \
            {
                'file_id': f'folder-{self.time_stamp}-{str(str(uuid4()))}',
                # 'file_real_name': f"{PR_NAME}_{TIME_STAMP}",
                'file_real_name': f"{str(os.path.basename(SAVE_FOLDER))}",
                'file_type': 'train_results_folder',
                # 'file_path':os.path.join(SAVE_FOLDER, f"{PR_NAME}_{TIME_STAMP}"),
                'file_path':str(os.path.join(SAVE_FOLDER,'train')),
                'file_comment': f"train_results_folder",
                'file_create_time': self.time_stamp
        }
        data_record(train_folder_record_msg)
        
        # 权重文件
        weights_record_msg = \
            {
            'file_id': f'weight-{self.time_stamp}-{str(str(uuid4()))}',
            'file_real_name': 'best.pt',
            'file_type': 'yolo_weight',
            # 'file_path':str(os.path.join(SAVE_FOLDER, f"{PR_NAME}_{TIME_STAMP}",'weights','best.pt')),
            'file_path':str(os.path.join(SAVE_FOLDER,'train','weights','best.pt')),
            'file_comment': 'yolo_weight',
            'file_create_time': self.time_stamp
        }
        data_record(weights_record_msg)
        

        torch.cuda.empty_cache()
        gc.collect()
        # return str(results.save_dir._str), str(weights_file_id)
        return str(train_folder_record_msg['file_id']), str(weights_record_msg['file_id'])
        
    def predict(self,args)->None:
        """预测大于等于0.25的目标
        Returns:
            _type_: _description_
        """
        weight_path = args['weight_path']
        test_images_path = args['files_path']
        save_folder = os.path.join(parent_folder, 'data', 'predict',f"Predict-{self.time_stamp}-{str(uuid4())[20:]}")
        os.makedirs(save_folder, exist_ok=True)
        detect_results = []
        yolo_model = YOLO(weight_path['file_path'])
        for test_image in test_images_path:
            predict_file_path = test_image['file_path']
            if not predict_file_path:
                detect_results.append(
                    {
                        'file_name': None,
                        'original_file_id': test_image['file_id'],
                        'predict_file_id': None,
                        'boxes': [],
                        'msg': '文件可能不存在'
                    }
                )
                continue
            if os.path.isfile(predict_file_path):
                try:              
                    results = yolo_model(test_image['file_path'], conf=0.25)
                    results_boxes =[]
                    for result in results:
                        boxes = result.boxes  # Boxes object for bounding box outputs
                        file_path = str(result.path)
                        file_name = os.path.basename(file_path)
                        for _box in boxes:
                            _conf = round(float(_box.conf.tolist()[0]),2)
                            _xywhn = _box.xywhn.tolist()[0]    
                            _cls = _box.cls.tolist()[0]
                            results_boxes.append(
                                {
                                    'class_name': int(_cls),
                                    'confidence': _conf,
                                    'x': _xywhn[0],
                                    'y': _xywhn[1],
                                    'w': _xywhn[2],
                                    'h': _xywhn[3]
                                }
                            ) 
                        
                        test_save_path = str(os.path.join(save_folder, file_name))
                        result.save(test_save_path)
                        record_msg = \
                        {
                            'file_id': f'predict-{self.time_stamp}-{str(uuid4())}',
                            'file_real_name': str(file_name),
                            'file_type': 'yolo_predict_image',
                            'file_path':test_save_path,
                            'file_comment': 'yolo_predict_image',
                            'file_create_time': self.time_stamp
                        }
                        data_record(record_msg)
                        
                        detect_results.append(
                            {
                                'file_name': file_name,
                                'predict_base64': f'data:image/png;base64,{TransferLocalImageFiles(test_save_path).toBase64()}',
                                'original_file_id': test_image['file_id'],
                                'predict_file_id': record_msg['file_id'],
                                'boxes': results_boxes,
                                'msg': f'检测到{len(results_boxes)}个目标'
                            }
                        )
                except:
                    continue
            elif os.path.isdir(predict_file_path):
                for img in os.listdir(predict_file_path):
                    try:
                        if img.endswith(('.jpg', '.png', '.jpeg')):
                            results = yolo_model(os.path.join(predict_file_path, img), conf=0.25)
                            results_boxes =[]
                            for result in results:
                                boxes = result.boxes  # Boxes object for bounding box outputs
                                file_path = str(result.path)
                                file_name = os.path.basename(file_path)
                                for _box in boxes:
                                    _conf = round(float(_box.conf.tolist()[0]),2)
                                    _xywhn = _box.xywhn.tolist()[0]    
                                    _cls = _box.cls.tolist()[0]
                                    results_boxes.append(
                                        {
                                            'class_name': int(_cls),
                                            'confidence': _conf,
                                            'x': _xywhn[0],
                                            'y': _xywhn[1],
                                            'w': _xywhn[2],
                                            'h': _xywhn[3]
                                        }
                                    ) 
                                
                                test_save_path = str(os.path.join(save_folder, file_name))
                                result.save(test_save_path)
                                record_msg = \
                                {
                                    'file_id': f'predict-{self.time_stamp}-{str(uuid4())}',
                                    'file_real_name': str(file_name),
                                    'file_type': 'yolo_predict_image',
                                    'file_path':test_save_path,
                                    'file_comment': 'yolo_predict_image',
                                    'file_create_time': self.time_stamp
                                }
                                data_record(record_msg)
                                
                                detect_results.append(
                                    {
                                        'file_name': file_name,
                                        'predict_base64': f'data:image/png;base64,{TransferLocalImageFiles(test_save_path).toBase64()}',
                                        'original_file_id': test_image['file_id'],
                                        'predict_file_id': record_msg['file_id'],
                                        'boxes': results_boxes,
                                        'msg': f'检测到{len(results_boxes)}个目标'
                                    }
                                )
                    except:
                        continue
            
            # 清除模型和缓存
            del yolo_model
            torch.cuda.empty_cache()
            # 写入日志
            with open(os.path.join(save_folder, 'predict_log.json'), 'a') as f:
                json_str = json.dumps(detect_results, indent=4, ensure_ascii=False)
                f.write(json_str + '\n')
            
            record_msg = \
                {
                    'file_id': f'log-{self.time_stamp}-{str(uuid4())}',
                    'file_real_name': 'predict_log.json',
                    'file_type': 'yolo_predict_log',
                    'file_path':os.path.join(save_folder, 'predict_log.json'),
                    'file_comment': 'yolo_predict_log',
                    'file_create_time': self.time_stamp
                }
            data_record(record_msg)
            
        # torch.cuda.empty_cache()
        # gc.collect()
        
        # 返回日志id 和 预测结果
        return str(record_msg['file_id']),detect_results
   
    def val(self,args):
        yaml_path = args['yaml_path']
        weight_path = args['weight_path']
        save_folder = os.path.join(parent_folder, 'data', 'val',f"Val-{self.time_stamp}-{str(uuid4())[20:]}")
        os.makedirs(save_folder, exist_ok=True)
        _confidences = args.get('conf',0.25)
        BATCH_SIZE = int(args.get('batch_size',8))
        IMAGE_SIZE = int(args.get('image_size',640))
        # DEVICE = int(args.get('device',0))
        DEVICE = args.get('device', None) or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        PLOTS = True
        try:
            _confidences = str(_confidences).split(',') 
            _confidences = [round(float(i),2) for i in _confidences]
        except:
            _confidences = [round(float(_confidences),2)]
        _confidences += [0.5, 0.75]
        yolo_model = YOLO(weight_path['file_path'])  # load a custom model
        result_boxes = [] # 保存每个阈值下的结果
        folder_boxes = [] # 保存验证文件的路径
        for _CONF in _confidences:
            metrics = yolo_model.val(
                        data=yaml_path['file_path'],
                        imgsz=IMAGE_SIZE,
                        batch=BATCH_SIZE,
                        project=str(save_folder),
                        name=str(_CONF),
                        device=DEVICE,
                        plots=PLOTS,
                        conf = _CONF
                    )
       
            # 验证文件夹
            folder_record_msg = \
                {
                    'file_id': f'val-{self.time_stamp}-{str(uuid4())}',
                    'file_real_name': str(_CONF),
                    'file_type': 'yolo_val_folder',
                    'file_path':os.path.join(save_folder,str(_CONF)),
                    'file_comment': 'yolo_val_folder',
                    'file_create_time': self.time_stamp
                }
            data_record(folder_record_msg)
            
            result_boxes.append({
                'conf': _CONF,
                'map50': round(float(metrics.box.map50),2),
                'map75': round(float(metrics.box.map75),2),
                'map50-95': round(float(metrics.box.map),2),
                'file_id': str(folder_record_msg['file_id']),
            })
            folder_boxes.append({
                'conf': _CONF,
                'file_id': folder_record_msg['file_id'],
            })
            
        val_results = {
            "yaml_id": yaml_path['file_id'],
            "weight_id": weight_path['file_id'],
            "result_boxes": result_boxes,
        }
        
        # 写入日志
        with open(os.path.join(save_folder, 'val_log.json'), 'a') as f:
            json_str = json.dumps(val_results, indent=4, ensure_ascii=False)
            f.write(json_str + '\n')
        
        # 日志文件
        log_record_msg = \
            {
                'file_id': f'log-{self.time_stamp}-{str(uuid4())}',
                'file_real_name': 'val_log.json',
                'file_type': 'yolo_val_log',
                'file_path':os.path.join(save_folder, 'val_log.json'),
                'file_comment': 'yolo_val_log',
                'file_create_time': self.time_stamp
            }
        data_record(log_record_msg)
        
        # torch.cuda.empty_cache()
        # gc.collect()
        
        # 返回日志id, 验证结果,验证文件夹id
        return str(log_record_msg['file_id']), val_results, folder_boxes


class TrainScripts:
    def __init__(self):
        self.time_stamp = str(time.strftime('%Y%m%d%H%M', time.localtime()))
    def get_train_point(self,train_comment='',is_create=False):
        folder_name = f"{self.time_stamp}{train_comment}-{str(uuid4())}"
        save_path = os.path.join(parent_folder, 'data', 'train',folder_name)
        os.makedirs(save_path, exist_ok=True)
        folder_info = {
                'file_id': f'folder-{self.time_stamp}-{str(uuid4())}',
                'file_real_name': str(folder_name),
                'file_type': 'train_folder(created)',
                'file_path':str(save_path),
                'file_comment': str(train_comment) if len(train_comment) > 1 else 'train_folder(created)', 
                'file_create_time':str(self.time_stamp) 
            }
        data_record(folder_info)
        # 如果需要创建文件夹
        folder_list = []
        folder_id = folder_info['file_id']
        yaml_file_id = None
        if is_create:
            folder_list = []
            for _dir in ['images', 'labels']:
                for _condition in ['train', 'val']:
                    _folder = f"{_dir}/{_condition}"
                    _folder_path = os.path.join(save_path, _folder)
                    os.makedirs(_folder_path, exist_ok=True)
                    folder_record_msg = \
                        {
                            'file_id': f"folder-{self.time_stamp}-{str(uuid4())}",
                            'file_real_name': str(_condition),
                            'file_type': f'{_folder}-folder',
                            'file_path':str(_folder_path),
                            'file_comment': f'{str(_folder)}-yolo-dataset',
                            'file_create_time': str(self.time_stamp)
                        }
                    data_record(folder_record_msg)
                    folder_id = folder_record_msg['file_id']
                    folder_list.append({
                        'folder_name': str(_folder),
                        'folder_id': folder_id,
                    })
            # yaml
            with open(os.path.join(save_path, 'train.yaml'), 'a') as f:
                f.write(f"path : {save_path}\n")
                f.write(f"train: {os.path.join(save_path, 'images/train')}\n")
                f.write(f"val: {os.path.join(save_path, 'images/val')}\n")
                f.write(f"names: \n")
                f.write(f"  0: anomaly\n")
                f.flush()
                yaml_record_msg = \
                    {
                        'file_id': f"yaml-{str(uuid4())}",
                        'file_real_name': 'train.yaml',
                        'file_type': f'yolo_yaml_config',
                        'file_path':str(os.path.join(save_path, 'train.yaml')),
                        'file_comment': f'{str(train_comment)}-yolo_yaml_config',
                        'file_create_time': str(self.time_stamp)
                    }
                yaml_file_id = yaml_record_msg['file_id']
                data_record(yaml_record_msg)
        
        # 生成训练训练结果所在的文件夹
        train_save_folder = os.path.join(parent_folder, 'data','results',f"{self.time_stamp}-Train-{str(uuid4())[20:]}")
        os.makedirs(train_save_folder,exist_ok=True)
        train_save_folder_msg = \
                    {
                        'file_id': f"folder-{self.time_stamp}-{str(uuid4())}",
                        'file_real_name': str(os.path.basename(train_save_folder)),
                        'file_type': f'train_results_folder',
                        'file_path':str(train_save_folder),
                        'file_comment': f'train_results_folder',
                        'file_create_time': str(self.time_stamp)
                    }
        train_save_folder_id = train_save_folder_msg['file_id']
        data_record(train_save_folder_msg)
          
        return {"folder_id": folder_id,"is_create":is_create,"save_folder":folder_list,"yaml_file_id":yaml_file_id,"train_save_folder_id":train_save_folder_id}       
            
            
    def upload_yolo_datasets(self,folder_id,train_data,file_ext,train_comment=''):
        # 获取文件扩展名
        # TODO
        # file_ext = os.path.splitext(train_data.filename)[1].lower()
        train_comment = str(train_comment) if len(train_comment) > 1 else 'train_data'
        folder_id = FilesID('folder_id',folder_id)
        folder_info = is_files_exist(folder_id) # 一个
        if not folder_info[0]['data'][0]['file_path']:
            return 'folder'
        save_folder = folder_info[0]['data'][0]['file_path']
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
        try:
            # 关闭文件描述符，将使用文件名
            os.close(temp_fd)
            # 保存上传的文件到临时位置
            train_data.save(temp_path)
            os.makedirs(save_folder, exist_ok=True)
            # 根据文件扩展名选择不同的解压方法
            if file_ext == '.zip':
                extract_zip(temp_path, save_folder)
            elif file_ext == '.rar':
                extract_rar(temp_path, save_folder)
            elif file_ext == '.7z':
                extract_7z(temp_path, save_folder)
        
        except Exception as e:
            return f'Extraction failed: {str(e)}'
        
        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
        # 找到save_folder内以.yaml结尾的文件
        yaml_files = glob.glob(os.path.join(save_folder, '*.yaml'))
        if not yaml_files or len(yaml_files)!=1:
            return 'yaml'
        
        update_yaml_config(
            yaml_file = yaml_files[0], 
            new_values = {
            "path": str(save_folder),
            "train": str(os.path.join(save_folder, 'images/train')),
            "val": str(os.path.join(save_folder, 'images/val'))
            }
        )
        #  yaml_data
        yaml_record_msg = \
                {
                    'file_id': f"yaml-{str(uuid4())}",
                    'file_real_name': str(os.path.basename(yaml_files[0])),
                    'file_type': 'yolo_yaml_config',
                    'file_path':str(yaml_files[0]),
                    'file_comment': f'{str(train_comment)}yolo_yaml_config',
                    'file_create_time': str(self.time_stamp)
                }
        data_record(yaml_record_msg)
        
        # 训练文件夹
        datasets_folder = os.path.dirname(yaml_files[0])
        folder_list = []
        for _dir in ['images', 'labels']:
            for _condition in ['train', 'val']:
                _folder = f"{_dir}/{_condition}"
                _folder_path = os.path.join(datasets_folder, _folder)
                folder_record_msg = \
                    {
                        'file_id': f"folder-{str(uuid4())}",
                        'file_real_name': str(_condition),
                        'file_type': f'{_folder}-folder',
                        'file_path':str(_folder_path),
                        'file_comment': f'{str(_folder)}-yolo-dataset',
                        'file_create_time': str(self.time_stamp)
                    }
                data_record(folder_record_msg)
                folder_list.append({
                    'folder_name': str(_folder),
                    'folder_id': folder_record_msg['file_id'],
                })
        
        return {"yaml_file_id": yaml_record_msg['file_id'],"save_folder":folder_list}





if __name__ == '__main__':
    # weight_path = '2dec00d4-e414-4136-bb92-33c4d0e91cf4'
    # test_images_path = ['image-202504112318-e8fdae37-c9ae-44ac-903b-8a86d097350a','image-202504112149-9a5b625c-bc25-4cdb-9cef-f95478cbb63a']
    # yolo_api = YoloAPI()
    # yolo_api.predict(weight_path,test_images_path=test_images_path,save_path='save_path')


    val = os.path.splitext('/Users/katsura/Documents/code/ultralytics/_api/data/predict/predict-202508150221-588-4d7d14fbb888/predict_results_conf0.25/003.png')
    print(val)