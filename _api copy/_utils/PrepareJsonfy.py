from datetime import datetime

def prepare_jsonfy(data):
    data.pop('_sa_instance_state', None)
    # 转换 datetime 对象为字符串
    if isinstance(data['create_time'], datetime):
        data['create_time'] = data['create_time'].isoformat()
    
    return data
