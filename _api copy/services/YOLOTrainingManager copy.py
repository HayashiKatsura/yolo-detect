import os
pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(pj_folder)    
from _project.mydata._code.yolo._train import standard_train
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json
import pandas as pd
import os
import time
import uuid
import yaml
import gc
import torch
import signal
import subprocess
import psutil
import zipfile
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import glob
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(project_path)
from uuid import uuid4
from _api._utils.DataRecord import data_record
from _api._utils.IsFilesExist import FilesID,is_files_exist,files_info
from _api.configuration.FilesRecordMapping import files_record_mapping,FilesType,table_1, table_2,table_3,table_4,table_5
from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.handle_db_error import handle_db_error

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# from loguru import logger as lgu
# lgu.remove()
# app = Flask(__name__)
# CORS(app)


def get_training_manager():
    return YOLOTrainingManager()

class CSVMonitorHandler(FileSystemEventHandler):
    def __init__(self, session_id, csv_path, manager):
        self.session_id = session_id
        self.csv_path = csv_path
        self.manager = manager
        self.last_row_count = 0
        self.last_modified = 0
        self.last_data_hash = None
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        logger.info(f"📁 开始监控CSV文件: {csv_path}")
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.csv_path:
            current_time = time.time()
            if current_time - self.last_modified < 0.5:
                return
            self.last_modified = current_time
            
            logger.info(f"检测到CSV文件变化: {event.src_path}")
            self.read_and_update()
    
    def read_and_update(self):
        """读取CSV并更新数据"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning(f"CSV文件不存在: {self.csv_path}")
                return
                
            df = None
            for attempt in range(5):
                try:
                    df = pd.read_csv(self.csv_path)
                    break
                except (pd.errors.EmptyDataError, FileNotFoundError, pd.errors.ParserError) as e:
                    if attempt < 4:
                        time.sleep(0.2)
                        continue
                    else:
                        logger.warning(f"CSV文件读取失败，尝试{attempt+1}次: {e}")
                        return
                except Exception as e:
                    logger.error(f"CSV读取异常: {e}")
                    return
            
            if df is None or len(df) == 0:
                logger.warning("CSV文件为空")
                return
                
            current_row_count = len(df)
            latest_row = df.iloc[-1]
            
            data_hash = hash(str(latest_row.to_dict()))
            if data_hash == self.last_data_hash and current_row_count == self.last_row_count:
                return
            
            if current_row_count == 1:
                logger.info(f"CSV列名: {list(df.columns)}")
            
            # 获取用户配置的总轮次
            session_info = self.manager.active_processes.get(self.session_id, {})
            configured_epochs = session_info.get('params', {}).get('epochs', 100)
            
            progress_data = self._parse_yolo_data(latest_row, current_row_count, df.columns, configured_epochs)
            
            if progress_data:
                if self.session_id in self.manager.active_processes:
                    self.manager.active_processes[self.session_id]['latest_data'] = progress_data
                    logger.info(f"📈 更新进度: Epoch {progress_data['epoch']}, mAP50: {progress_data['metrics']['mAP50']:.3f}")
                
                self.last_row_count = current_row_count
                self.last_data_hash = data_hash
                
        except Exception as e:
            logger.error(f"❌ CSV读取错误: {e}")
    
    def _parse_yolo_data(self, row, total_rows, columns, configured_epochs):
        """解析YOLO数据 - 修复轮次显示"""
        try:
            logger.debug(f"可用列: {list(columns)}")
            
            epoch = 0
            for col in ['epoch', 'Epoch']:
                if col in row:
                    epoch = int(float(row[col])) + 1
                    break
            
            if epoch == 0:
                logger.warning("未找到epoch列")
                return None
            
            train_box_loss = self._get_value_by_possible_names(row, [
                'train/box_loss', 'box_loss', 'train_box_loss', 'Box'
            ])
            train_obj_loss = self._get_value_by_possible_names(row, [
                'train/obj_loss', 'obj_loss', 'train_obj_loss', 'Objectness', 'train/cls_loss'
            ])
            train_cls_loss = self._get_value_by_possible_names(row, [
                'train/cls_loss', 'cls_loss', 'train_cls_loss', 'Classification'
            ])
            
            val_box_loss = self._get_value_by_possible_names(row, [
                'val/box_loss', 'val_box_loss', 'val/Box'
            ])
            val_obj_loss = self._get_value_by_possible_names(row, [
                'val/obj_loss', 'val_obj_loss', 'val/Objectness', 'val/cls_loss'
            ])
            val_cls_loss = self._get_value_by_possible_names(row, [
                'val/cls_loss', 'val_cls_loss', 'val/Classification'
            ])
            
            precision = self._get_value_by_possible_names(row, [
                'metrics/precision(B)', 'metrics/precision', 'precision', 'Precision', 'P'
            ])
            recall = self._get_value_by_possible_names(row, [
                'metrics/recall(B)', 'metrics/recall', 'recall', 'Recall', 'R'
            ])
            map50 = self._get_value_by_possible_names(row, [
                'metrics/mAP50(B)', 'metrics/mAP_0.5', 'metrics/mAP@0.5', 'mAP50', 'mAP_0.5', 'mAP@0.5'
            ])
            map50_95 = self._get_value_by_possible_names(row, [
                'metrics/mAP50-95(B)', 'metrics/mAP_0.5:0.95', 'metrics/mAP@0.5:0.95', 'mAP50-95', 'mAP_0.5:0.95'
            ])
            
            lr = self._get_value_by_possible_names(row, [
                'lr/pg0', 'lr/pg1', 'lr/pg2', 'learning_rate', 'lr', 'LR'
            ])
            
            result = {
                "epoch": epoch,
                "total_epochs": configured_epochs,  # 使用用户配置的轮次
                "train_losses": {
                    "box_loss": train_box_loss,
                    "obj_loss": train_obj_loss,
                    "cls_loss": train_cls_loss,
                    "total_loss": train_box_loss + train_obj_loss + train_cls_loss
                },
                "val_losses": {
                    "box_loss": val_box_loss,
                    "obj_loss": val_obj_loss,
                    "cls_loss": val_cls_loss,
                    "total_loss": val_box_loss + val_obj_loss + val_cls_loss
                },
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "mAP50": map50,
                    "mAP50_95": map50_95
                },
                "learning_rate": lr,
                "timestamp": time.time()
            }
            
            logger.debug(f"解析结果: Epoch {epoch}/{configured_epochs}, Box: {train_box_loss:.4f}, mAP50: {map50:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ 数据解析错误: {e}")
            return None
    
    def _get_value_by_possible_names(self, row, possible_names):
        """根据可能的列名获取值"""
        for name in possible_names:
            if name in row and pd.notna(row[name]):
                try:
                    return float(row[name])
                except (ValueError, TypeError):
                    continue
        return 0.0

class YOLOTrainingManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(YOLOTrainingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self.active_processes = {}
        self.csv_monitors = {}
        self.file_observers = {}
        self.training_threads = {}
        self.stop_events = {}
        self.csv_handlers = {}
        self.training_processes = {}  # 只记录训练子进程
        self._initialized = True
        self.time_stamp =str(time.strftime("%Y%m%d%H%M", time.localtime()))
        self.save_folder_id = ''
        # self.save_folder_id = f"train-folder-{self.time_stamp}-{str(uuid4())[20:]}" # 训练文件夹存储id
        
        
        logger.info("YOLOTrainingManager 初始化完成")
    
    def stop_training(self, session_id):
        """精确停止训练 - 修复版"""
        if session_id not in self.active_processes:
            return False, "训练会话不存在"
        
        try:
            logger.info(f"🛑 开始停止训练: {session_id}")
            
            if session_id in self.stop_events:
                self.stop_events[session_id].set()
                logger.info(f"✅ 设置停止事件: {session_id}")
            
            self._stop_csv_monitoring(session_id)
            success = self._force_stop_training_process(session_id)
            
            if session_id in self.training_threads:
                training_thread = self.training_threads[session_id]
                training_thread.join(timeout=5.0)
                
                if training_thread.is_alive():
                    logger.warning(f"⚠️ 训练线程 {session_id} 仍在运行")
                else:
                    logger.info(f"✅ 训练线程 {session_id} 已停止")
            
            if session_id in self.active_processes:
                self.active_processes[session_id]['status'] = 'stopped'
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._cleanup_session_resources(session_id)
            
            message = "训练已精确停止" if success else "训练停止信号已发送"
            return True, message
            
        except Exception as e:
            logger.error(f"停止训练出错: {e}")
            return False, f"停止失败: {str(e)}"
    
    def _force_stop_training_process(self, session_id):
        """精确停止训练进程 - 修复版，避免误杀Flask服务"""
        try:
            current_pid = os.getpid()
            flask_process = psutil.Process(current_pid)
            
            # 方法1: 如果有记录的训练子进程，直接终止
            if session_id in self.training_processes:
                process = self.training_processes[session_id]
                if process and process.poll() is None:
                    logger.info(f"🎯 终止已记录的训练子进程: PID {process.pid}")
                    try:
                        # 终止子进程及其所有子进程
                        parent = psutil.Process(process.pid)
                        children = parent.children(recursive=True)
                        
                        # 先终止子进程
                        for child in children:
                            try:
                                child.terminate()
                                logger.info(f"🔥 终止子进程: PID {child.pid}")
                            except psutil.NoSuchProcess:
                                pass
                        
                        # 再终止主进程
                        parent.terminate()
                        
                        # 等待进程结束
                        try:
                            parent.wait(timeout=3)
                            logger.info(f"✅ 训练进程已正常终止")
                            return True
                        except psutil.TimeoutExpired:
                            # 强制杀死
                            for child in children:
                                try:
                                    child.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            parent.kill()
                            logger.info(f"⚡ 强制杀死训练进程")
                            return True
                            
                    except psutil.NoSuchProcess:
                        logger.info(f"ℹ️ 训练进程已不存在")
                        return True
                    except Exception as e:
                        logger.error(f"终止训练进程时出错: {e}")
            
            # 方法2: 查找特定的训练进程（更精确的条件）
            terminated_count = 0
            
            # 只查找Flask进程的直接子进程
            try:
                flask_children = flask_process.children(recursive=True)
                
                for child in flask_children:
                    try:
                        cmdline = ' '.join(child.cmdline())
                        
                        # 更精确的匹配条件：
                        # 1. 包含yolo训练相关关键词
                        # 2. 是Python进程
                        # 3. 不是Flask主进程
                        # 4. 包含train相关参数
                        if (child.name() in ['python', 'python3'] and 
                            any(keyword in cmdline.lower() for keyword in ['yolo', 'ultralytics']) and
                            'train' in cmdline.lower() and
                            'app.py' not in cmdline and  # 排除Flask主进程
                            'flask' not in cmdline.lower() and  # 排除Flask相关进程
                            child.pid != current_pid):
                            
                            logger.info(f"🎯 找到训练子进程: PID {child.pid}")
                            logger.info(f"🔍 进程命令: {cmdline[:200]}")
                            
                            child.terminate()
                            terminated_count += 1
                            
                            try:
                                child.wait(timeout=3)
                            except psutil.TimeoutExpired:
                                try:
                                    child.kill()
                                    logger.info(f"⚡ 强制杀死进程: PID {child.pid}")
                                except psutil.NoSuchProcess:
                                    pass
                                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
            except Exception as e:
                logger.error(f"查找子进程时出错: {e}")
            
            if terminated_count > 0:
                logger.info(f"✅ 精确终止了 {terminated_count} 个训练进程")
                return True
            else:
                logger.info("ℹ️ 未找到需要终止的训练进程")
                return False
                
        except Exception as e:
            logger.error(f"精确停止进程失败: {e}")
            return False
    
    def _run_training(self, session_id, params, stop_event):
        """运行训练 - 改进子进程管理"""
        try:
            logger.info(f"🏃 开始训练会话: {session_id}")
            self.active_processes[session_id]['status'] = 'running'
            
            threading.Thread(
                target=self._enhanced_csv_monitoring,
                args=(session_id, params),
                daemon=True
            ).start()
            
            # 使用子进程运行训练
            actual_save_dir = self._run_training_subprocess(session_id, params, stop_event)
            
            if actual_save_dir:
                self.active_processes[session_id]['save_dir'] = actual_save_dir
                logger.info(f"📁 实际保存目录: {actual_save_dir}")
            
            if stop_event.is_set():
                self.active_processes[session_id]['status'] = 'stopped'
                logger.info(f"🛑 训练被停止: {session_id}")
            else:
                self.active_processes[session_id]['status'] = 'completed'
                logger.info(f"✅ 训练完成: {session_id}")
                
                # if actual_save_dir:
                #     self._auto_zip_training_results(actual_save_dir)
            
        except InterruptedError:
            logger.info(f"🛑 训练被中断: {session_id}")
            if session_id in self.active_processes:
                self.active_processes[session_id]['status'] = 'stopped'
        except Exception as e:
            logger.error(f"❌ 训练错误 {session_id}: {e}")
            if session_id in self.active_processes:
                self.active_processes[session_id]['status'] = 'error'
                self.active_processes[session_id]['error'] = str(e)
        finally:
            self._cleanup_session_resources(session_id)
    
    def _run_training_subprocess(self, session_id, params, stop_event):
        """在子进程中运行训练 - 改进版"""
        try:
            # 直接调用训练函数，而不是创建脚本
            # 这样可以更好地控制进程
            actual_save_dir = None
            
            def training_worker():
                nonlocal actual_save_dir
                try:
                    params['stop_event'] = stop_event
                    actual_save_dir = standard_train(
                        config_yaml_path=params['config_yaml_path'],
                        data_yaml=params['data_yaml'],
                        save_path= params['save_path'],
                        batch_size=params['batch_size'],
                        epochs = params['epochs'],
                        image_size=params['image_size'],
                        learning_rate=params['learning_rate'],
                        device=params['device'],
                        rtd_yolo=params['rtd_yolo'],
                        session_id=self.save_folder_id
                    )
                except Exception as e:
                    logger.error(f"训练子进程出错: {e}")
                    raise
            
            # 在线程中运行训练（而不是子进程）
            # 这样更容易控制和停止
            training_thread = threading.Thread(target=training_worker, daemon=True)
            training_thread.start()
            
            # 监控训练线程
            while training_thread.is_alive():
                if stop_event.is_set():
                    logger.info(f"🛑 检测到停止信号")
                    break
                training_thread.join(timeout=1)
            
            # 等待线程完成
            training_thread.join(timeout=5)
            
            return actual_save_dir
                
        except Exception as e:
            logger.error(f"训练执行失败: {e}")
            raise
    
    def start_training(self, config):
        """启动YOLO训练"""
        self.save_folder_id = f"train-folder-{self.time_stamp}-{str(uuid4())[20:]}" # 训练文件夹存储id
        session_id = str(self.save_folder_id)
        
        try:
            # if not self._validate_config(config):
            #     return None, False, "配置验证失败", None
            
            training_params = self._prepare_training_params(config)
            
            # 训练文件夹
            # record_msg = \
            #         {
            #             "file_id":str(self.save_folder_id),
            #             "file_folder_id":str(self.save_folder_id),
            #             "file_path":str(training_params['save_path']),
            #             "file_folder_path":str(training_params['save_path']),
            #             "file_real_name":str(training_params['name']), # 前端传入的项目名称
            #             "file_type":'folder',
            #             'file_comment':'trained-weights',
            #             'dataset_id':str(config.get('trainData')),
            #             "is_detected":False,
            #             "file_create_time": self.time_stamp,
            #             "session_id":str(session_id),
            #         }
            # data_record(record_msg,fieldnames=table_5(),save_path=files_record_mapping()[str(FilesType.weights)][1])
            record_msg = WeightTable(
                    file_id = str(self.save_folder_id),
                    file_path = str(training_params['save_path']),
                    folder_path = str(training_params['save_path']),
                    file_name = str(training_params['name']),
                    dataset_id = str(config.get('trainData')),
                    session_id = str(session_id),
                    train_log = str(os.path.join(project_path, '_api/logs/train',f"{self.save_folder_id}.log"))
            )
            db.session.add(record_msg)
            db.session.commit()
            
            logger.info(f"🚀 启动训练，参数: {training_params}")

            stop_event = threading.Event()
            self.stop_events[session_id] = stop_event
            
            self.active_processes[session_id] = {
                'status': 'starting',
                'config': config,
                'params': training_params,
                'start_time': time.time(),
                'latest_data': None,
                'save_dir': None,
                'stop_event': stop_event,
                'csv_path': None
            }
            
            training_thread = threading.Thread(
                target=self._run_training,
                args=(session_id, training_params, stop_event),
                daemon=True
            )
            self.training_threads[session_id] = training_thread
            training_thread.start()
            
            save_dir = training_params['save_path']
            self.active_processes[session_id]['save_dir'] = save_dir
            
            return session_id, True, "训练启动成功", save_dir,self.save_folder_id
            
        except Exception as e:
            # files_info("file_id", self.save_folder_id, delete_row=True,record_path=files_record_mapping()[str(FilesType.weights)][1])
            
            logger.error(f"❌ 启动失败详情: {e}")
            if session_id in self.stop_events:
                del self.stop_events[session_id]
            if session_id in self.active_processes:
                del self.active_processes[session_id]
            return None, False, f"训练启动失败: {str(e)}", None, None
    
    def _stop_csv_monitoring(self, session_id):
        """停止CSV监控"""
        try:
            if session_id in self.file_observers:
                observer = self.file_observers[session_id]
                if observer.is_alive():
                    observer.stop()
                    observer.join(timeout=5.0)
                del self.file_observers[session_id]
                logger.info(f"✅ 停止文件监控器: {session_id}")
            
            if session_id in self.csv_handlers:
                del self.csv_handlers[session_id]
            
            if session_id in self.csv_monitors:
                del self.csv_monitors[session_id]
                
        except Exception as e:
            logger.error(f"停止CSV监控失败: {e}")
    
    def _cleanup_session_resources(self, session_id):
        """清理会话资源"""
        try:
            resources_to_clean = [
                (self.stop_events, "停止事件"),
                (self.training_threads, "训练线程"),
                (self.training_processes, "训练进程"),
                (self.csv_handlers, "CSV处理器"),
                (self.csv_monitors, "CSV监控"),
                (self.file_observers, "文件观察器")
            ]
            
            for resource_dict, resource_name in resources_to_clean:
                if session_id in resource_dict:
                    del resource_dict[session_id]
                    logger.info(f"🧹 清理{resource_name}: {session_id}")
            
            logger.info(f"✅ 会话资源清理完成: {session_id}")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    
    def get_training_status(self, session_id=None):
        """获取训练状态"""
        if session_id:
            raw_data = self.active_processes.get(session_id)
            if not raw_data:
                return None
            
            return {
                'status': raw_data.get('status'),
                'start_time': raw_data.get('start_time'),
                'csv_path': raw_data.get('csv_path'),
                'save_dir': raw_data.get('save_dir'),
                'latest_data': raw_data.get('latest_data'),
                'error': raw_data.get('error'),
                'config': raw_data.get('config', {}),
                'params': {k: v for k, v in raw_data.get('params', {}).items() 
                          if k != 'stop_event' and not callable(v)}
            }
        else:
            result = {}
            for sid, raw_data in self.active_processes.items():
                result[sid] = {
                    'status': raw_data.get('status'),
                    'start_time': raw_data.get('start_time'),
                    'csv_path': raw_data.get('csv_path'),
                    'save_dir': raw_data.get('save_dir'),
                    'latest_data': raw_data.get('latest_data'),
                    'error': raw_data.get('error')
                }
            return result
    
    def get_latest_progress(self, session_id):
        """获取最新的训练进度"""
        if session_id in self.active_processes:
            return self.active_processes[session_id].get('latest_data')
        return None
    
    def _validate_config(self, config):
        """验证训练配置"""
        try:
            required_fields = ['data_yaml', 'config_yaml_path']
            for field in required_fields:
                if not config.get(field):
                    logger.error(f"❌ 缺少必要参数: {field}")
                    return False
            
            files_to_check = [
                ('data_yaml', config.get('data_yaml')),
                ('config_yaml_path', config.get('config_yaml_path'))
            ]
            
            for field_name, file_path in files_to_check:
                if file_path and not os.path.exists(file_path):
                    logger.error(f"❌ 文件不存在 {field_name}: {file_path}")
                    return False
            
            logger.info("✅ 配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 配置验证错误: {e}")
            return False
    
    def _prepare_training_params(self, config):
        """训练参数"""
        params = {
            'config_yaml_path': config.get('version', 'yolov8n.pt'),
            'data_yaml_id': config.get('trainData'),
            # 'weight_path': config.get('weight_path'),
            # 'pr_name': config.get('name', ''),
            # 'desc': config.get('desc', ''),
            'save_path': str(os.path.join(project_path, '_api/data', 'train',self.save_folder_id)),
            'batch_size': int(config.get('batch', 8)),
            'epochs': int(config.get('epoch', 100)),
            'image_size': int(config.get('size', 640)),
            'learning_rate': float(config.get('lr', 0.01)),
            'device': config.get('device', 'cpu'),
            # 'freeze': config.get('freeze',None),
            'rtd_yolo': config.get('type', 'yolo'),
            # 'data_yaml':str(files_info('file_id',config.get('trainData'),record_path=files_record_mapping()[str(FilesType.datasets)][1])['yaml_path']),
            'name':config.get('name', str(self.save_folder_id)),
            'data_yaml':str(DatasetTable.query.filter_by(file_id=config.get('trainData')).first().yaml_path),
        }
        
        # if str(params['device']).lower() == 'cpu':
        #     params['device'] = 'cpu'
        # else:
        #     try:
        #         params['device'] = int(params['device'])
        #     except:
        #         params['device'] = 0
        
        return params
    
    def _enhanced_csv_monitoring(self, session_id, params):
        """增强的CSV监控"""
        logger.info(f"🔍 开始增强CSV监控: {session_id}")
        
        save_path = params.get('save_path', './runs/detect')
        possible_base_dirs = [
            save_path,
            './runs/detect', 
            './runs', 
            './runs/train'
        ]
        
        max_wait_time = 600
        check_interval = 3
        
        for elapsed in range(0, max_wait_time, check_interval):
            if session_id not in self.active_processes:
                logger.info(f"会话 {session_id} 已结束，停止CSV监控")
                break
            
            found_csv = None
            
            for base_dir in possible_base_dirs:
                if not os.path.exists(base_dir):
                    continue
                
                try:
                    csv_files = glob.glob(os.path.join(base_dir, "**/results.csv"), recursive=True)
                    
                    if csv_files:
                        latest_csv = max(csv_files, key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0)
                        
                        if time.time() - os.path.getmtime(latest_csv) < 300:
                            found_csv = latest_csv
                            logger.info(f"✅ 找到CSV: {found_csv}")
                            break
                            
                except Exception as e:
                    logger.warning(f"搜索 {base_dir} 时出错: {e}")
            
            if found_csv:
                self.active_processes[session_id]['csv_path'] = found_csv
                self._start_csv_monitoring(session_id, found_csv)
                return
            
            time.sleep(check_interval)
            if elapsed % 15 == 0:
                logger.info(f"⏳ 搜索CSV文件中... ({elapsed}秒)")
        
        logger.warning(f"❌ 等待{max_wait_time}秒后仍未找到CSV文件")
    
    def _start_csv_monitoring(self, session_id, csv_path):
        """开始监控CSV文件"""
        try:
            self._stop_csv_monitoring(session_id)
            
            if not os.path.exists(csv_path):
                logger.warning(f"CSV文件不存在: {csv_path}")
                return False
            
            observer = Observer()
            handler = CSVMonitorHandler(session_id, csv_path, self)
            
            watch_dir = str(Path(csv_path).parent)
            
            try:
                observer.schedule(handler, watch_dir, recursive=False)
                observer.start()
                
                self.file_observers[session_id] = observer
                self.csv_handlers[session_id] = handler
                self.csv_monitors[session_id] = csv_path
                
                logger.info(f"✅ 开始监控CSV文件: {csv_path}")
                
                handler.read_and_update()
                self._start_polling_backup(session_id, handler)
                
                return True
                
            except OSError as e:
                if "inotify watch limit reached" in str(e):
                    logger.warning(f"⚠️ inotify监控限制达到，切换到轮询模式: {session_id}")
                    observer.stop()
                    return self._start_polling_mode(session_id, csv_path)
                else:
                    raise
            
        except Exception as e:
            logger.error(f"❌ 启动CSV监控失败: {e}")
            return False
    
    def _start_polling_mode(self, session_id, csv_path):
        """轮询模式监控CSV文件"""
        try:
            handler = CSVMonitorHandler(session_id, csv_path, self)
            self.csv_handlers[session_id] = handler
            self.csv_monitors[session_id] = csv_path
            
            handler.read_and_update()
            self._start_polling_backup(session_id, handler)
            
            logger.info(f"✅ 启动轮询模式监控: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动轮询模式失败: {e}")
            return False
    
    def _start_polling_backup(self, session_id, handler):
        """启动轮询备份机制"""
        def polling_worker():
            last_check_time = 0
            check_interval = 3
            
            while session_id in self.active_processes:
                try:
                    current_time = time.time()
                    if current_time - last_check_time >= check_interval:
                        status = self.active_processes[session_id].get('status')
                        if status == 'running':
                            handler.read_and_update()
                        elif status in ['completed', 'error', 'stopped']:
                            break
                        last_check_time = current_time
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"轮询监控错误: {e}")
                    time.sleep(5)
            
            logger.info(f"停止轮询监控: {session_id}")
        
        polling_thread = threading.Thread(target=polling_worker, daemon=True)
        polling_thread.start()
    
    def _auto_zip_training_results(self, save_dir):
        """自动打包训练结果"""
        try:
            if not os.path.exists(save_dir):
                logger.warning(f"保存目录不存在，无法打包: {save_dir}")
                return
            
            training_folder = Path(save_dir)
            folder_name = training_folder.name
            parent_dir = training_folder.parent
            zip_path = parent_dir / f"{folder_name}.zip"
            
            logger.info(f"🗜️ 开始打包训练结果: {save_dir}")
            logger.info(f"📦 压缩包路径: {zip_path}")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(save_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, parent_dir)
                        zipf.write(file_path, arcname)
                        logger.debug(f"📄 添加文件: {arcname}")
            
            zip_size = os.path.getsize(zip_path)
            zip_size_mb = zip_size / (1024 * 1024)
            
            logger.info(f"✅ 打包完成!")
            logger.info(f"📦 压缩包大小: {zip_size_mb:.2f} MB")
            
            print(f"=== 训练完成，压缩包路径 ===")
            print(f"压缩包位置: {zip_path}")
            print(f"压缩包大小: {zip_size_mb:.2f} MB")
            print(f"原始文件夹: {save_dir}")
            print("================================")
            
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"❌ 打包失败: {e}")
            return None