from . import api_model

from flask import Flask, request, jsonify
from ...services.YOLOTrainingManager import get_training_manager
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import pandas as pd
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLO, RTDETR
from _api.services.YoloAPI import YoloAPI
from _api._utils.ResultsResponse.NoStandardResponse import NoStandardResponse
from _api.entity.ResponseCode import ResponseCode

# from _project.mydata._code.yolo._test import standard_test

# project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# import sys
# sys.path.append(project_path)

# 配置日志
# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
from loguru import logger
logger.remove()

"""
yolo相关
"""
@api_model.route('/training/start', methods=['POST'])
def start_training():
    """启动训练API"""
    try:
        config = request.json
        logger.info(f"📋 收到训练配置: {config}")
        
        manager = get_training_manager()
        session_id, success, message, save_dir,save_folder_id = manager.start_training(config)
        
        return jsonify({
            'success': success,
            'session_id': session_id,
            'message': message,
            'save_dir': save_dir,
            'save_folder_id': save_folder_id
        })
    except Exception as e:
        logger.error(f"❌ 启动训练失败: {e}")
        return jsonify({
            'success': False,
            'message': f'启动失败: {str(e)}'
        }), 500

@api_model.route('/training/stop/<session_id>', methods=['POST'])
def stop_training(session_id):
    """停止训练并重置状态"""
    try:
        manager = get_training_manager()
        success, message = manager.stop_training(session_id)
        
        # 自动重置状态
        if session_id in manager.active_processes:
            del manager.active_processes[session_id]
            logger.info(f"🧹 自动重置会话状态: {session_id}")
        
        return jsonify({
            'success': success,
            'message': message,
            'auto_reset': True
        })
        
    except Exception as e:
        logger.error(f"停止训练失败: {e}")
        return jsonify({
            'success': False,
            'message': f'停止失败: {str(e)}'
        }), 500

@api_model.route('/training/status', methods=['GET'])
@api_model.route('/training/status/<session_id>', methods=['GET'])
def get_training_status(session_id=None):
    """获取训练状态API"""
    try:
        manager = get_training_manager()
        status = manager.get_training_status(session_id)
        
        return jsonify({
            'success': True,
            'data': status
        })
        
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取状态失败: {str(e)}'
        }), 500

@api_model.route('/training/progress/<session_id>', methods=['GET'])
def get_training_progress(session_id):
    """获取训练进度API"""
    try:
        manager = get_training_manager()
        progress = manager.get_latest_progress(session_id)
        
        session_info = manager.get_training_status(session_id)
        debug_info = {
            'session_exists': session_id in manager.active_processes,
            'csv_path': session_info.get('csv_path') if session_info else None,
            'status': session_info.get('status') if session_info else None
        }
        
        return jsonify({
            'success': True,
            'data': progress,
            'debug': debug_info
        })
        
    except Exception as e:
        logger.error(f"❌ 获取进度失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取进度失败: {str(e)}'
        }), 500

@api_model.route('/health', methods=['GET'])
def health_check():
    """健康检查API"""
    try:
        manager = get_training_manager()
        return jsonify({
            'success': True,
            'message': 'YOLO训练服务器运行正常',
            'active_sessions': len(manager.active_processes),
            'sessions': list(manager.active_processes.keys())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'健康检查失败: {str(e)}'
        }), 500

@api_model.route('/debug/<session_id>', methods=['GET'])
def debug_session(session_id):
    """调试会话信息"""
    try:
        training_manager = get_training_manager()
        session_info = training_manager.get_training_status(session_id)
        
        debug_data = {
            'session_id': session_id,
            'session_exists': session_id in training_manager.active_processes,
            'all_sessions': list(training_manager.active_processes.keys()),
            'session_info': session_info,
            'csv_monitors': training_manager.csv_monitors,
        }
        
        if session_info:
            debug_data['session_details'] = {
                'status': session_info.get('status'),
                'start_time': session_info.get('start_time'),
                'csv_path': session_info.get('csv_path'),
                'latest_data': session_info.get('latest_data'),
                'save_dir': session_info.get('save_dir')
            }
            
            csv_path = session_info.get('csv_path')
            if csv_path:
                debug_data['csv_info'] = {
                    'path': csv_path,
                    'exists': os.path.exists(csv_path),
                    'size': os.path.getsize(csv_path) if os.path.exists(csv_path) else 0,
                    'modified_time': os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0
                }
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        debug_data['csv_content'] = {
                            'rows': len(df),
                            'columns': list(df.columns),
                            'last_few_rows': df.tail(3).to_dict('records') if len(df) > 0 else []
                        }
                    except Exception as e:
                        debug_data['csv_error'] = str(e)
            
            save_dir = session_info.get('save_dir')
            if save_dir:
                debug_data['save_dir_info'] = {
                    'path': save_dir,
                    'exists': os.path.exists(save_dir),
                    'files': []
                }
                
                if os.path.exists(save_dir):
                    try:
                        files = os.listdir(save_dir)
                        debug_data['save_dir_info']['files'] = files
                        csv_files = [f for f in files if f.endswith('.csv')]
                        debug_data['save_dir_info']['csv_files'] = csv_files
                    except Exception as e:
                        debug_data['save_dir_info']['error'] = str(e)
        
        return jsonify({
            'success': True,
            'debug_data': debug_data
        })
        
    except Exception as e:
        logger.error(f"调试失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_model.route('/force_csv_scan/<session_id>', methods=['POST'])
def force_csv_scan(session_id):
    """强制扫描CSV文件"""
    try:
        training_manager = get_training_manager()
        if session_id not in training_manager.active_processes:
            return jsonify({
                'success': False,
                'message': '会话不存在'
            }), 404
        
        session_info = training_manager.active_processes[session_id]
        save_dir = session_info.get('save_dir')
        
        if save_dir:
            logger.info(f"🔍 强制扫描CSV文件,目录: {save_dir}")
            
            threading.Thread(
                target=training_manager._enhanced_csv_monitoring,
                args=(session_id, session_info.get('params', {})),
                daemon=True
            ).start()
            
            return jsonify({
                'success': True,
                'message': '已重新扫描CSV文件',
                'csv_path': session_info.get('csv_path')
            })
        else:
            return jsonify({
                'success': False,
                'message': '没有保存目录信息'
            })
            
    except Exception as e:
        logger.error(f"CSV扫描失败: {e}")
        return jsonify({
            'success': False,
            'message': f'扫描失败: {str(e)}'
        }), 500

@api_model.route('/training/zip/<session_id>', methods=['POST'])
def zip_training_results(session_id):
    """手动压缩训练结果"""
    try:
        manager = get_training_manager()
        
        if session_id not in manager.active_processes:
            return jsonify({
                'success': False,
                'message': '训练会话不存在'
            }), 404
        
        session_info = manager.active_processes[session_id]
        save_dir = session_info.get('save_dir')
        
        if not save_dir:
            return jsonify({
                'success': False,
                'message': '没有找到训练保存目录'
            }), 400
        
        zip_path = manager._auto_zip_training_results(save_dir)
        
        if zip_path:
            return jsonify({
                'success': True,
                'message': '压缩完成',
                'zip_path': zip_path,
                'save_dir': save_dir
            })
        else:
            return jsonify({
                'success': False,
                'message': '压缩失败'
            }), 500
            
    except Exception as e:
        logger.error(f"手动压缩失败: {e}")
        return jsonify({
            'success': False,
            'message': f'压缩失败: {str(e)}'
        }), 500


@api_model.route('/detect_file', methods=['GET'], strict_slashes=False)
def detect_image():
    """
    预测图像
    """
    weight_id = request.args.get('weight_id')
    image_id = request.args.get('image_id')
    images_id = request.args.get('images_id',None)
    camera = request.args.get('camera',False)
    if images_id and images_id!='null':
        image_id = images_id
    conf = float(request.args.get('conf'))
    results = YoloAPI(weight_id,conf,camera=camera).detect_file(image_id)
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())
          
@api_model.route("/progress/<file_id>", methods=["GET"])
def video_detect_progress(file_id):
    """
    视频检测进度
    """
    results = YoloAPI(None).detect_progress(file_id)
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())

@api_model.route('/val_weight/<weight_id>', methods=['GET'], strict_slashes=False)
def val_weight(weight_id):
    """
    验证权重
    """
    dataYamlId = request.args.get('dataYamlId')
    conf = float(request.args.get('conf'))
    results = YoloAPI(weight_id,conf).val_weight(dataYamlId)
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())
 


@api_model.route('/valid_params/<file_id>', methods=['GET'], strict_slashes=False)
def valid_params(file_id):
    """
    参数校验
    """
    results = YoloAPI(None).validate(file_id)
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())

 
 
 
 
 
 
# @api_model.route("/api/progress/<task_id>", methods=["GET"]) # TODO
# def video_detect_progress(task_id):
#     """
#     视频检测进度
#     """
#     progress = processing_progress.get(task_id, -1)
#     return jsonify({"code": 200, "progress": progress})

 



# # TODO ZWX 新增
# import base64
# import io
# import cv2
# import numpy as np
# from PIL import Image
# import time
# import tempfile
# from datetime import timedelta
# import csv
# import uuid
# from flask import Flask, request, jsonify,send_from_directory,make_response



# # add摄像头实时检测
# # 全局模型存储,避免重复加载

# model = None
# target_classes = [0, 1 , 2]  # 目标类别

# def init_model():
#     global model
#     if model is None:
#         model_path = "/home/panxiang/coding/kweilx/ultralytics/zwx.pt"
#         # 第一步：先验证文件是否存在
#         if not os.path.exists(model_path):
#             raise Exception(f"模型文件不存在！路径：{model_path}")
        
#         model = YOLO(model_path)  # 关键：添加device参数
#         print(f"模型加载成功,使用设备: {model.device}")  
# # 启动时初始化模型
# init_model()
# @api_model.route('/detect_camera', methods=['POST'])
# def detect_camera():
#     # 处理摄像头实时检测请求(使用绝对路径加载的固定模型
#     try:
#                 # 日志1：确认接口被调用
#         print("\n=== 收到 /zjut/detect_camera 请求 ===")
#         # 口罩，安全帽
#         model_path = "/home/panxiang/coding/kweilx/ultralytics/safety_helmet.pt"

#         # 加载模型
#         print("开始加载模型...")
#         print(f"模型加载成功,使用设备: {model.device}")  
#         # model = YOLO(model_path,device=0)
#         print("模型加载完成")
#         # 执行预测
#         print("开始执行预测...")
#         # results = model.predict(image_path, classes=[0,2], save=True)
#         print("预测执行完成")
#         # 1. 获取请求数据
#         data = request.json
#         print("请求数据是否存在:", "是" if data else "否")  # 日志2
#         if not data or 'frame' not in data:
#             print("错误：缺少frame参数")  # 日志3
#             return jsonify({
#                 'status': 'error',
#                 'message': '缺少必要参数: frame'
#             }), 400
#         # 2. 解析参数(原代码基础上补充日志）
#         print("请求数据中的所有字段:", data.keys())  # 新增日志：查看data里是否有frame
#         if 'frame' not in data:
#             print("错误：缺少frame参数")
#             return jsonify({'status':'error','message':'缺少必要参数: frame'}),400

#         frame_data = data['frame']
#         conf_threshold = data.get('conf', 0.25)
#         # 新增日志：查看frame长度(有效Base64至少几千字符,空字符串或短字符串都是无效的）
#         print(f"frame参数长度: {len(frame_data)} 字符")  
#                 # 获取请求数据
#         data = request.json
#         if not data or 'frame' not in data:
#             return jsonify({
#                 'status': 'error',
#                 'message': '缺少必要参数: frame'
#             }), 400

#         # 解析参数
#         frame_data = data['frame']  # Base64格式图像
#         conf_threshold = data.get('conf', 0.25)  # 置信度阈值,默认0.25

#         # 解析Base64图像
#         try:
#             # 去除Base64前缀
#             if ',' in frame_data:
#                 frame_data = frame_data.split(',')[1]
#                 print("已去除Base64前缀,处理后长度:", len(frame_data))
            
#             # 解码Base64
#             img_bytes = base64.b64decode(frame_data)
#             print(f"Base64解码成功,得到 {len(img_bytes)} 字节的图像数据")


#             img = Image.open(io.BytesIO(img_bytes))
#             print(f"图像解析成功：尺寸 {img.size},格式 {img.format}")
#             img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             print("已转换为OpenCV格式,准备推理")
#         except Exception as e:
#             return jsonify({
#                 'status': 'error',
#                 'message': f'图像解析失败: {str(e)}'
#             }), 400

#         # 模型推理
#         print(f"\n===== 开始模型推理 =====")
#         print(f"模型是否加载: {'是' if model is not None else '否(模型为None！）'}")  # 关键：确认模型已加载
#         print(f"输入图像尺寸: {img_cv.shape}(高x宽x通道数,应与1280x720对应）")
#         print(f"置信度阈值: {conf_threshold}")
#         print(f"使用设备: {model.device}")  # 验证推理时的设备
#         start_time = time.time()
#         results = model(img_cv, conf=conf_threshold,classes=target_classes)
#         infer_time = round((time.time() - start_time) * 1000, 2)  # 推理时间(ms)
#         print(f"推理耗时: {infer_time} ms")

#           # 验证推理结果结构
#         print(f"推理返回结果数量: {len(results)}(正常应为1,对应单帧图像）")
#         if len(results) == 0:
#             print("错误：模型返回空结果(未处理图像）")
#             return jsonify({
#                 'status': 'error',
#                 'message': '模型未返回任何推理结果'
#             }), 500

#         result = results[0]
#         print(f"第一帧结果包含的目标数量: {len(result.boxes)}")  # 关键：是否有检测到目标
#         if len(result.boxes) > 0:
#             box = result.boxes[0]
#             print(f"第一个目标信息:")
#             print(f"  类别ID: {int(box.cls[0])}")
#             print(f"  类别名称: {model.names[int(box.cls[0])]}")  # 验证类别名称是否存在
#             print(f"  置信度: {float(box.conf[0]):.2f}")
#             print(f"  坐标(x1,y1,x2,y2): {box.xyxy[0].tolist()}")
#         # 处理检测结果
#         detections = []
#         for result in results:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 cls_name = model.names[int(box.cls[0])]
#                 confidence = float(box.conf[0])
#                 detections.append({
#                     'class': cls_name,
#                     'confidence': round(confidence, 2),
#                     'bbox': [round(x1), round(y1), round(x2), round(y2)],
#                     'inference_time': infer_time
#                 })
#         print(f"最终提取的检测结果数量: {len(detections)}")  # 确认结果列表非空
#         print(f"准备返回的响应数据: {detections}")  # 打印完整结果,确认格式正确

#         return jsonify({
#             'status': 'success',
#             'detections': detections,
#             'timestamp': int(time.time() * 1000)
#         })

#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': f'检测过程出错: {str(e)}'
#         }), 500
    
    
    
# 新增,处理视频文件识别请求
# @api_model.route('/detect_video', methods=['POST'])
# def detect_video():
#     # 处理视频文件识别请求
#     print("\n=== 收到 /zjut/detect_video 请求 ===")
#     try:
#         # 获取请求参数
#         data = request.json
#         if not data or 'video_data' not in data:
#             return jsonify({
#                 'status': 'error',
#                 'message': '缺少必要参数: video_data'
#             }), 400

#         # 解析参数
#         video_base64 = data['video_data']
#         conf_threshold = data.get('conf', 0.25)
#         frame_interval = data.get('frame_interval', 10)  # 每隔多少帧处理一次
#         max_frames = data.get('max_frames', 1000)  # 最大处理帧数,防止内存溢出

#         # 解码Base64视频数据
#         try:
#             if ',' in video_base64:
#                 video_base64 = video_base64.split(',')[1]
#             video_bytes = base64.b64decode(video_base64)
#             logger.info(f"视频数据解码成功,大小: {len(video_bytes)} bytes")
#         except Exception as e:
#             return jsonify({
#                 'status': 'error',
#                 'message': f'视频解码失败: {str(e)}'
#             }), 400

#         # 创建临时文件存储视频
#         with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
#             temp_file.write(video_bytes)
#             temp_video_path = temp_file.name

#         # 初始化视频捕获
#         cap = cv2.VideoCapture(temp_video_path)
#         if not cap.isOpened():
#             os.unlink(temp_video_path)
#             return jsonify({
#                 'status': 'error',
#                 'message': '无法打开视频文件'
#             }), 400

#         # 获取视频基本信息
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         logger.info(f"开始处理视频: 总帧数={frame_count}, FPS={fps}, 分辨率={width}x{height}")

#         # 处理视频帧
#         results = []
#         frame_idx = 0
#         processed_frames = 0
        
#         while cap.isOpened() and processed_frames < max_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # 按间隔处理帧
#             if frame_idx % frame_interval == 0:
#                 start_time = time.time()
                
#                 # 转换为RGB格式(YOLO模型要求）
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
#                 # 模型推理
#                 detections = model(frame_rgb, conf=conf_threshold, classes=target_classes)
                
#                 # 处理检测结果
#                 frame_results = []
#                 for det in detections:
#                     for box in det.boxes:
#                         x1, y1, x2, y2 = box.xyxy[0].tolist()
#                         cls_name = model.names[int(box.cls[0])]
#                         confidence = float(box.conf[0])
#                         frame_results.append({
#                             'class': cls_name,
#                             'confidence': round(confidence, 2),
#                             'bbox': [round(x1), round(y1), round(x2), round(y2)]
#                         })
                
#                 # 计算时间戳(毫秒）
#                 timestamp = int((frame_idx / fps) * 1000)
                
#                 results.append({
#                     'frame_index': frame_idx,
#                     'timestamp': timestamp,
#                     'detections': frame_results,
#                     'inference_time': round((time.time() - start_time) * 1000, 2)
#                 })
                
#                 processed_frames += 1
#                 logger.debug(f"处理帧 {frame_idx}/{frame_count}, 检测到 {len(frame_results)} 个目标")

#             frame_idx += 1

#         # 释放资源
#         cap.release()
#         os.unlink(temp_video_path)  # 删除临时文件

#         logger.info(f"视频处理完成,共处理 {processed_frames} 帧")

#         return jsonify({
#             'status': 'success',
#             'video_info': {
#                 'total_frames': frame_count,
#                 'fps': fps,
#                 'resolution': f"{width}x{height}",
#                 'processed_frames': processed_frames
#             },
#             'results': results,
#             'timestamp': int(time.time() * 1000)
#         })

#     except Exception as e:
#         logger.error(f"视频识别出错: {str(e)}")
#         return jsonify({
#             'status': 'error',
#             'message': f'视频处理过程出错: {str(e)}'
#         }), 500

# def load_model():
#     global model
#     if model is None:
#         model_path = "/home/panxiang/coding/kweilx/ultralytics/zwx.pt"
#         # 第一步：先验证文件是否存在
#         if not os.path.exists(model_path):
#             raise Exception(f"模型文件不存在！路径：{model_path}")
        
#         model = YOLO(model_path)  # 关键：添加device参数
#         print(f"模型加载成功,使用设备: {model.device}")  
# # 启动时初始化模型
# load_model()

# # 文件夹配置
# UPLOAD_FOLDER = "/home/panxiang/coding/kweilx/ultralytics/uploads"
# OUTPUT_FOLDER = "/home/panxiang/coding/kweilx/ultralytics/outputs"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# processing_progress = {}   # 任务进度存储
# # 新增：打印完整路径(关键）
# print("原始视频文件夹完整路径：", os.path.abspath(UPLOAD_FOLDER))
# print("结果视频文件夹完整路径：", os.path.abspath(OUTPUT_FOLDER))



# def generate_target_csv(task_id, target_frames, fps, output_folder):
    
#     # 同一目标(同类别+连续帧）的检测段,仅保留一个中间时间写入CSV
    
#     # 步骤1：按帧索引排序,确保帧顺序正确
#     target_frames.sort(key=lambda x: x["frame_idx"])
#     if not target_frames:
#         return

#     # 步骤2：识别“同一目标(类别+连续帧）”的检测段
#     target_segments = []  # 存储目标段：[{start_idx, end_idx, class, start_ms, end_ms, mid_frame}, ...]
    
#     # 初始化第一个目标段(按帧内每个目标分别初始化）
#     first_frame = target_frames[0]
#     for det in first_frame["detections"]:
#         target_segments.append({
#             "class": det["class"],  # 目标类别(核心：按类别区分同一目标）
#             "start_idx": first_frame["frame_idx"],
#             "end_idx": first_frame["frame_idx"],
#             "start_ms": first_frame["timestamp_ms"],
#             "end_ms": first_frame["timestamp_ms"],
#             "frames": [first_frame]  # 存储该目标段的所有帧
#         })

#     # 遍历后续帧,扩展或新增目标段
#     for frame in target_frames[1:]:
#         current_frame_idx = frame["frame_idx"]
#         current_frame_dets = {det["class"]: det for det in frame["detections"]}  # 按类别存储当前帧目标

#         # 1. 处理已有目标段：判断当前帧是否有同类别目标且帧连续
#         updated_segments = []
#         for seg in target_segments:
#             seg_class = seg["class"]
#             # 判定：当前帧有同类别目标 + 帧连续(当前帧索引=段结束帧+1）
#             if seg_class in current_frame_dets and current_frame_idx == seg["end_idx"] + 1:
#                 # 扩展目标段：更新结束帧、结束时间、添加当前帧
#                 seg["end_idx"] = current_frame_idx
#                 seg["end_ms"] = frame["timestamp_ms"]
#                 seg["frames"].append(frame)
#                 updated_segments.append(seg)
#                 # 从当前帧目标中移除已匹配的类别(避免重复处理）
#                 del current_frame_dets[seg_class]
#             else:
#                 # 目标段不连续或无同类别目标,保留原段
#                 updated_segments.append(seg)
#         target_segments = updated_segments

#         # 2. 处理当前帧中未匹配的新目标(新增目标段）
#         for det_class, det in current_frame_dets.items():
#             target_segments.append({
#                 "class": det_class,
#                 "start_idx": current_frame_idx,
#                 "end_idx": current_frame_idx,
#                 "start_ms": frame["timestamp_ms"],
#                 "end_ms": frame["timestamp_ms"],
#                 "frames": [frame]
#             })

#     # 步骤3：过滤持续时间 < 1秒(1000毫秒）的目标段
#     filtered_segments = []
#     for seg in target_segments:
#         seg_duration_ms = seg["end_ms"] - seg["start_ms"]
#         if seg_duration_ms >= 1000:  # 仅保留持续≥1秒的目标段
#             filtered_segments.append(seg)
#     if not filtered_segments:
#         print(f"任务 {task_id}：无持续≥1秒的目标段,不生成CSV")
#         return

#     # 步骤4：时间戳格式转换(毫秒→分:秒.毫秒）
#     def format_time(ms):
#         td = timedelta(milliseconds=ms)
#         minutes = int(td.total_seconds() // 60)
#         seconds = int(td.total_seconds() % 60)
#         ms_remain = td.microseconds // 1000
#         return f"{minutes:02d}:{seconds:02d}.{ms_remain:03d}"

#     # 步骤5：生成CSV(同一目标段仅写入一个中间时间）
#     csv_filename = f"{task_id}_target_timeline.csv"
#     csv_path = os.path.join(output_folder, csv_filename)

#     with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
#         fieldnames = [
#             "目标段ID", "目标类别", "中间帧时间", 
#             "中间帧索引", "目标置信度", "持续时间(秒)"
#         ]
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()

#         # 遍历过滤后的目标段,每个段仅写入一条中间时间记录
#         for seg_id, seg in enumerate(filtered_segments, 1):
#             # 计算当前目标段的中间帧
#             mid_idx = (seg["start_idx"] + seg["end_idx"]) // 2
#             # 找到中间帧数据(取中间帧中该类别的目标置信度）
#             mid_frame = next(f for f in seg["frames"] if f["frame_idx"] == mid_idx)
#             mid_det = next(d for d in mid_frame["detections"] if d["class"] == seg["class"])
#             # 计算中间时间和持续时间
#             mid_time = format_time(mid_frame["timestamp_ms"])
#             duration_sec = round((seg["end_ms"] - seg["start_ms"]) / 1000, 2)

#             # 同一目标段仅写入一条记录(含唯一中间时间）
#             writer.writerow({
#                 "目标段ID": seg_id,
#                 "目标类别": seg["class"],
#                 "中间帧时间": mid_time,  # 同一目标段仅一个中间时间
#                 "中间帧索引": mid_idx,
#                 "目标置信度": round(mid_det["confidence"], 2),
#                 "持续时间(秒)": duration_sec
#             })

#     print(f"任务 {task_id}：CSV生成完成,共{len(filtered_segments)}个目标段(每段一个中间时间）,路径：{csv_path}")
    
# def process_video(video_path, output_path, task_id):
#     target_frames = []  # 记录含目标(置信度≥0.3）的帧
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width, height = int(cap.get(3)), int(cap.get(4))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   
    
#     # 初始化视频写入器(H.264编码,分辨率偶数处理）
#     fourcc = cv2.VideoWriter_fourcc(*"avc1")
#     width = width if width % 2 == 0 else width - 1
#     height = height if height % 2 == 0 else height - 1
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
        
#         # 调整帧大小(与输出视频一致）
#         if frame.shape[1] != width or frame.shape[0] != height:
#             frame = cv2.resize(frame, (width, height))
        
#         # 模型检测(置信度≥0.3,过滤低置信目标）
#         results = model(frame, conf=0.3)
#         annotated_frame = results[0].plot()
        
#         # 记录含目标(置信度≥0.3）的帧信息
#         frame_detections = []
#         for box in results[0].boxes:
#             det_conf = float(box.conf[0])
#             if det_conf >= 0.3:  # 仅保留置信度≥0.3的目标
#                 frame_detections.append({
#                     "class": model.names[int(box.cls[0])],
#                     "confidence": det_conf
#                 })
#         if frame_detections:  # 仅当帧含有效目标时记录
#             target_frames.append({
#                 "frame_idx": frame_count,
#                 "timestamp_ms": int((frame_count / fps) * 1000),
#                 "detections": frame_detections
#             })

#         # 写入标注视频 + 更新处理进度
#         out.write(annotated_frame)
#         frame_count += 1
#         processing_progress[task_id] = min(int((frame_count / total_frames) * 100), 100)
    
#     # 视频处理完成后,生成CSV(确保仅执行一次）
#     if target_frames:
#         generate_target_csv(task_id, target_frames, fps, OUTPUT_FOLDER)
#     else:
#         print(f"任务 {task_id}：未检测到置信度≥0.3的目标,不生成CSV")
    
#     # 释放资源(避免文件占用）
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     # 接口部分(不变,确保前端能正常上传、查进度、取结果）
# @api_model.route("/api/upload", methods=["POST"]) # TODO
# def upload_video():
#     if "video" not in request.files:
#         return jsonify({"code": 400, "msg": "未选择视频文件"}), 400
    
#     file = request.files["video"]
#     task_id = str(uuid.uuid4())
#     input_path = os.path.join(UPLOAD_FOLDER, f"{task_id}.mp4")
#     output_path = os.path.join(OUTPUT_FOLDER, f"{task_id}.mp4")
    
#     file.save(input_path)
#     processing_progress[task_id] = 0
#     threading.Thread(
#         target=process_video, args=(input_path, output_path, task_id), daemon=True
#     ).start()
    
#     return jsonify({"code": 200, "msg": "上传成功,开始检测", "task_id": task_id})



# @api_model.route("/api/result/<task_id>", methods=["GET"]) # TODO
# def get_result(task_id):
#     # 定义结果视频路径(结合输出文件夹和task_id）
#     result_video_path = os.path.join(OUTPUT_FOLDER, f"{task_id}.mp4")
#     # 构建视频路径
#     result_path = os.path.join(OUTPUT_FOLDER, f"{task_id}.mp4")

#     # 新增：构建CSV文件路径
#     csv_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_target_timeline.csv")
    
#     # 检查文件是否存在
#     if not os.path.exists(result_path):
#         return jsonify({"code": 404, "msg": "结果视频未生成"}), 404
    
#     # 新增：判断CSV文件是否存在
#     csv_exists = os.path.exists(csv_path)
#     csv_url = f"/api/download_csv/{task_id}" if csv_exists else None  # 后续可新增下载接口

#     # 检查文件大小(排除空文件）
#     file_size = os.path.getsize(result_path)
#     if file_size < 1024:  # 小于1KB视为无效文件
#         return jsonify({"code": 500, "msg": "视频文件损坏"}), 500
    
#     # 处理Range请求(断点续传核心逻辑）
#     range_header = request.headers.get('Range', None)
#     if range_header:
#         # 解析Range头部(格式示例：bytes=0-1023）
#         try:
#             # 提取起始和结束字节
#             range_part = range_header.split('=')[1]
#             start_str, end_str = range_part.split('-')
#             start = int(start_str) if start_str else 0
#             end = int(end_str) if end_str else file_size - 1
            
#             # 确保结束位置不超过文件大小
#             end = min(end, file_size - 1)
#             content_length = end - start + 1
            
#             # 读取视频片段
#             with open(result_path, 'rb') as f:
#                 f.seek(start)
#                 video_data = f.read(content_length)
            
#             # 构建206部分内容响应
#             response = make_response(video_data)
#             response.status_code = 206  # 部分内容状态码
#             response.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
#             response.headers['Content-Length'] = str(content_length)
            
#         except Exception as e:
#             # 解析Range失败时返回完整文件
#             response = make_response(send_from_directory(OUTPUT_FOLDER, f"{task_id}.mp4"))
#     else:
#         # 无Range请求时返回完整文件
#         response = make_response(send_from_directory(OUTPUT_FOLDER, f"{task_id}.mp4"))
    

#         # 新增：在响应头或JSON中返回CSV信息(若用JSON返回,需调整接口逻辑,示例如下）
#     # 注：若保持原有“返回视频流”逻辑,可在响应头中添加CSV路径,或单独新增下载接口
#     # 此处示例为“返回JSON+视频流”的混合方式(实际可根据需求调整）
#     # (若仅需返回视频流,可删除此部分,仅在需要时提供CSV下载）
#     response_data = {
#         "code": 200,
#         "msg": "success",
#         "video_url": f"/api/result/{task_id}",
#         "csv_exists": csv_exists,
#         "csv_url": csv_url
#     }

#      # 此处保持原有视频流返回,仅在日志中打印CSV信息
#     print(f"任务 {task_id}：结果视频路径：{result_video_path},CSV路径：{csv_path if csv_exists else '无'}")
#     # 关键响应头设置
#     response.headers['Content-Type'] = 'video/mp4'  # 固定视频MIME类型
#     response.headers['Accept-Ranges'] = 'bytes'     # 告知浏览器支持断点续传
#     response.headers['Access-Control-Allow-Origin'] = '*'  # 允许跨域
    
#     return response


# @api_model.route("/api/download_csv/<task_id>", methods=["GET"]) # TODO
# def download_csv(task_id):
# # CSV文件路径(与process_video中生成的路径一致）
#     csv_path = os.path.join(OUTPUT_FOLDER, f"{task_id}_target_timeline.csv")
#     print("csv_path",csv_path,)
#     # 检查文件是否存在
#     if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 10:  # 排除空文件
#         return jsonify({"code": 404, "msg": "CSV文件不存在"}), 404
    
#     # 发送文件并设置下载响应头
#     response = send_from_directory(
#         OUTPUT_FOLDER, 
#         f"{task_id}_target_timeline.csv",
#         as_attachment=True,  # 强制下载
#         download_name=f"{task_id}_目标时间记录.csv"  # 下载文件名
#     )
#     response.headers["Access-Control-Allow-Origin"] = "*"  # 允许跨域
#     return response

