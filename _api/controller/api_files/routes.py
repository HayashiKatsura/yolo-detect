import requests
from flask import jsonify,request,send_file,Response
from _api._utils.ResultsResponse.NoStandardResponse import NoStandardResponse
from _api.entity.ResponseCode import ResponseCode
from _api.services.UploadFiles import UploadFilesLocalStorage
from _api.services.GetFiles import GetFilesLocalStorage
from _api.services.DownloadOrDeleteFIles import DowenloadOrDeleteFilesLocalStorage
from _api._utils.ReadTable import ReadLocalTable
from _api._utils.FilesReader import FilesReader

from _api.configuration.log_api_call import log_api_call

from werkzeug.utils import secure_filename
import os
import mimetypes

from . import api_files


@api_files.route('/upload_file/<folder_id>', methods=['POST'], strict_slashes=False)
@log_api_call
def upload_file(folder_id):
    """
    上传文件，不接收指定文件夹
    """
    files_data = request.files.getlist('files',None)
    camera = request.form.get('camera',None)
    results =  UploadFilesLocalStorage(files_data,folder_id,camera=camera)._upload_file()
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())



@api_files.route('/show_storage/<file_type>', methods=['GET'], strict_slashes=False)
@log_api_call
def show_storage(file_type):
    """
    获取上传文件信息，对象为本地csv表格
    """
    results =  ReadLocalTable(file_type).show_files_local_table()
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())
    
 
@api_files.route('/show_text/<file_id>', methods=['GET'], strict_slashes=False)
@log_api_call
def show_text(file_id):
    """
    渲染文本文件
    """
    results =  FilesReader(file_id).read_yaml()
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())
     
    
    
@api_files.route('/show_image/<file_id>/<tag>', methods=['GET'], strict_slashes=False)
@api_files.route('/show_image/<file_id>', methods=['GET'], strict_slashes=False)  # 无 tag 的路由
@log_api_call
def show_image(file_id,tag=None):
    """
    渲染图像,base64
    """
    range_header = request.headers.get('Range', None) # 视频相关
    results =  GetFilesLocalStorage().show_images(file_id,range=range_header,tag=tag)
    if videos_path:=results.get("videos_path",None):
        return send_file(videos_path, mimetype="video/mp4")
    elif videos_data:=results.get("videos_data",None):
        response =  Response(videos_data,
                        status=206,
                        mimetype='video/mp4',
                        direct_passthrough=True)
        response.headers.add('Content-Range', results.get("Content-Range"))
        response.headers.add('Accept-Ranges', results.get("Accept-Ranges"))
        response.headers.add('Content-Length', results.get("Content-Length"))

        return response
        
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())



@api_files.route('/delete_file/<file_id>', methods=['DELETE'], strict_slashes=False)
@log_api_call
def delete_file(file_id):
    """
    删除文件
    """
    results =  DowenloadOrDeleteFilesLocalStorage(file_id).delete_files()
    return jsonify(NoStandardResponse(ResponseCode.success.value, "success", data=results).get_response_body())



@api_files.route('/download_file/<file_id>', methods=['GET'], strict_slashes=False)
@log_api_call
def download_file(file_id):
    """
    文件下载
    """
    try:
        detect_id = request.args.get('detect_id',None)  # 下载检测的结果
        val = request.args.get('val',None) # 下载验证的结果
        camera = request.args.get('camera',None) # 下载采集的验证结果
        dataset_example = request.args.get('dataset_example',None) # 下载示范数据集
        train_log = request.args.get('train_log',None) # 下载训练日志
        seesion_id = request.args.get('seesion_id',None) # 下载的会话id
        train_id = request.args.get('train_id',None) # 下载的训练id
        is_detected = request.args.get('is_detected',None) # 是否下载检测结果
        only_video_csv = request.args.get('only_video_csv',None)
        
        
        download_infomation = DowenloadOrDeleteFilesLocalStorage(file_id,
                                                                 detect_id,
                                                                 val,camera=camera,
                                                                 dataset_example=dataset_example,
                                                                 train_log = train_log,
                                                                 seesion_id=seesion_id,
                                                                 train_id=train_id,
                                                                 is_detected=is_detected,
                                                                 only_video_csv = only_video_csv
                                                                 ).download_files()
        if not download_infomation:
            return jsonify(NoStandardResponse(ResponseCode.unknown_error.value, "下载失败").get_response_body())
        path_or_file, mimetype, download_name,need_delete = \
        download_infomation['path_or_file'], download_infomation['mimetype'], download_infomation['download_name'], download_infomation['need_delete']
        
        if not need_delete:
            # 发送文件
            return send_file(
                path_or_file=path_or_file,
                mimetype=mimetype,
                as_attachment=True,  # 强制下载而不是在浏览器中打开
                download_name=download_name  # Flask 2.0+ 使用 download_name
            )
        else:
            # 发送完成后删除
            def cleanup():
                # 下载完成后清理临时文件
                try:
                    os.remove(path_or_file)
                except:
                    pass
            
            response = send_file(
                path_or_file,
                mimetype='application/zip',
                as_attachment=True,
                download_name=download_name
            )
            
            # 注册清理函数（在响应完成后执行）
            response.call_on_close(cleanup)
            return response
        
    except Exception as e:
        return jsonify(NoStandardResponse(ResponseCode.unknown_error.value, str(e)).get_response_body())