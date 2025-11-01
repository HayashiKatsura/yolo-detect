import os
# 获取当前项目所在路径
PJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(PJ_ROOT)
import shutil
from _api.configuration.RedisConfig import redis_config
from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from sqlmodel import create_engine, text  
from sqlalchemy.exc import SQLAlchemyError
import re

datasets = [
    "datasets",
    "images",
    "videos",
    "weights",
    "trains",
    "predictions",
    "validations"
]

def init_storage():
    """
    删除存储路径下的所有文件
    """
    
    # 删除文件夹
    for item in datasets:
        shutil.rmtree(os.path.join(PJ_ROOT,'_api/data' ,item)) if os.path.exists(os.path.join(PJ_ROOT,'_api/data' ,item)) else None
    shutil.rmtree(os.path.join(PJ_ROOT,'_api/logs/train')) if os.path.exists(os.path.join(PJ_ROOT,'_api/logs/train')) else None
    
    # 清空redis
    redis_config.get_client().flushdb()
    
    # 清空数据库
    sql_file = os.path.join(PJ_ROOT,'_api/configuration/sql/create_table.sql')
    try:
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
    except Exception as e:
        print(f"读取 .sql 文件失败：{str(e)}")
        return
    
    
    # 移除注释
    sql_script = re.sub(r'--.*$', '', sql_script, flags=re.MULTILINE)
    sql_script = re.sub(r'/\*.*?\*/', '', sql_script, flags=re.DOTALL)
    
    # 分割 SQL 语句
    sql_commands = [cmd.strip() for cmd in sql_script.split(';') if cmd.strip()]
    
    with SQL_ENGINE.begin() as conn:
        for i, command in enumerate(sql_commands, 1):
            try:
                # 使用 text() 包装 SQL 语句
                conn.execute(text(command))
                print(f"成功执行第 {i} 条语句")
            except Exception as e:
                print(f"第 {i} 条语句执行出错: {e}")
                print(f"SQL: {command[:100]}...")  # 只打印前100个字符
                # 根据需要决定是否继续执行
                # raise  # 如果想在出错时停止，取消注释这行

if __name__ == '__main__':
    init_storage()