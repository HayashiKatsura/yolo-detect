from enum import Enum

# 响应码
class ResponseCode(Enum):
    """_summary_
    Args:
        Enum (_type_): 
    Returns:
        200: 成功
        255: 空数据
        404: 找不到文件
        500: 未知错误
    """
    success = 200
    empty_data = 255
    not_found = 404 
    unknown_error = 500

    def __str__(self):
        return self.value