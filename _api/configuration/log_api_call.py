import csv
import os
import time
from datetime import datetime
from functools import wraps
from flask import request

pj_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys    
sys.path.append(pj_folder)
LOG_FILE = os.path.join(pj_folder, '_api', 'logs', 'api_log.csv')

def log_api_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ip = request.remote_addr
        method_name = func.__name__

        try:
            response = func(*args, **kwargs)
            if hasattr(response, 'status_code'):
                code = response.status_code
            else:
                code = 200  # 默认成功状态码
        except Exception as e:
            code = 500
            raise e
        finally:
            time_cost = round(time.time() - start_time, 3)

            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['date', 'ip', 'method', 'code', 'time_cost'])
                writer.writerow([date, ip, method_name, code, time_cost])

        return response

    return wrapper
