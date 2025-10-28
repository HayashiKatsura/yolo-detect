from sqlalchemy.exc import SQLAlchemyError
from flask import Flask, request, jsonify
from _api.entity.SQLModels import db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_db_error(func):
    """数据库错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"数据库错误: {e}")
        except Exception as e:
            db.session.rollback()
            logger.error(f"未知错误: {e}")
    wrapper.__name__ = func.__name__
    return wrapper