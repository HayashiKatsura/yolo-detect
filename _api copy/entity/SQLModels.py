from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import validates
import time
TIME_STAMP = str(time.strftime("%Y%m%d%H%M", time.localtime()))
# 初始化数据库
db = SQLAlchemy()

def create_db(app):
    """创建数据库表"""
    with app.app_context():
        db.create_all()

class FileTable(db.Model):
    __tablename__ = 'files'

    file_id = db.Column(db.String(255), primary_key=True, unique=True, nullable=False, comment='唯一文件Id')
    file_path = db.Column(db.String(255), comment='实际文件路径')
    folder_path = db.Column(db.String(255), comment='实际文件所在文件夹路径')
    file_name = db.Column(db.String(255), comment='真实文件名')
    type = db.Column(db.String(255), comment='文件类型')
    comment = db.Column(db.String(255), comment='文件描述')
    is_detected = db.Column(db.String(255), default=None, comment='检测信息')
    create_time = db.Column(db.String(255), default=TIME_STAMP, comment='创建时间')
    update_time = db.Column(db.String(255), onupdate=TIME_STAMP, comment='更新时间')
    is_delete = db.Column(db.Boolean, default=False, comment='是否删除')


class DetectionTable(db.Model):
    __tablename__ = 'detections'

    file_id = db.Column(db.String(255), primary_key=True, unique=True, nullable=False, comment='文件id')
    weight_id = db.Column(db.String(255), comment='权重id')
    details = db.Column(db.JSON, comment='检测信息细节')
    create_time = db.Column(db.String(255), default=TIME_STAMP, comment='创建时间')
    update_time = db.Column(db.String(255), onupdate=TIME_STAMP, comment='更新时间')


class DatasetTable(db.Model):
    __tablename__ = 'datasets'

    file_id = db.Column(db.String(255), primary_key=True, unique=True, nullable=False, comment='唯一文件id')
    file_path = db.Column(db.String(255), comment='文件路径')
    file_name = db.Column(db.String(255), comment='文件名称')
    yaml_path = db.Column(db.String(255), comment='配置文件路径')
    images_folder = db.Column(db.String(255), comment='图像文件夹路径')
    labels_folder = db.Column(db.String(255), comment='标签文件夹路径')
    train_counts = db.Column(db.Integer, default=0,comment='训练集数量')
    last_train_counts = db.Column(db.Integer, default=0,comment='上一次训练集数量')
    val_counts = db.Column(db.Integer, default=0,comment='验证集数量')
    last_val_counts = db.Column(db.Integer, default=0,comment='上一次验证集数量')
    create_time = db.Column(db.String(255), default=TIME_STAMP, comment='创建时间')
    update_time = db.Column(db.String(255), onupdate=TIME_STAMP, comment='更新时间')
    is_delete = db.Column(db.Boolean, default=False, comment='是否删除')

class WeightTable(db.Model):
    __tablename__ = 'weights'

    file_id = db.Column(db.String(255), primary_key=True, unique=True,nullable=False, comment='唯一文件id')
    file_path = db.Column(db.String(255), comment='文件路径')
    folder_path = db.Column(db.String(255), comment='文件夹路径')
    file_name = db.Column(db.String(255), comment='文件名称')
    dataset_id = db.Column(db.String(255), default=None,comment='数据集id')
    is_validated = db.Column(db.String(255), default=None,comment='是否通过验证')
    session_id = db.Column(db.String(255), default=None,comment='会话id')
    train_log = db.Column(db.String(255), default=None,comment='训练日志')
    create_time = db.Column(db.String(255), default=TIME_STAMP, comment='创建时间')
    update_time = db.Column(db.String(255), onupdate=TIME_STAMP, comment='更新时间')
    is_delete = db.Column(db.Boolean, default=False, comment='是否删除')


