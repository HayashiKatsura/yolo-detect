import os
from enum import Enum
from flask import Flask, Response, jsonify,request
from flask_cors import CORS
from _api._utils.ResultsResponse.NoStandardResponse import NoStandardResponse
from _api._utils.CreateTable import CreateTable
from _api._utils.GetFiles import check_file_types,files_real_path
from _api.services.YoloAPI import YoloAPI, TrainScripts
from _api._utils.IsFilesExist import FilesID,is_files_exist
from _api._utils.UploadFiles import upload_files_scripts
from _api.controller.api_files.routes import api_files  
from _api.controller.api_model.routes import api_model  
from _api.entity.SQLModels import db, FileTable, DetectionTable, DatasetTable, WeightTable, create_db
from _api.configuration.SQLConfig import SQLConfig
parent_folder = os.path.dirname(os.path.abspath(__file__))


os.environ['NO_PROXY'] = '127.0.0.1'
def create_app():
    app = Flask(__name__)
    
    # 最大上传文件大小为 500 MB
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
    
    # 注册蓝图
    app.register_blueprint(api_files, url_prefix='/zjut')
    app.register_blueprint(api_model, url_prefix='/zjut')
    
    # 加载 SQL 配置
    app.config.from_object(SQLConfig)
    
    # 初始化数据库
    db.init_app(app) 
    
    CORS(app)
    return app

app = create_app()



if __name__ == '__main__':
    # 建表
    # CreateTable(delete_before_create=False)._create_tables()
    CreateTable(delete_before_create=True)._create_tables()
    
    app.run(host='0.0.0.0', port=5130, debug=True,use_reloader=False)
