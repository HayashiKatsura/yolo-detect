from flask import jsonify
from typing import Any, Dict, Optional, Union
from datetime import datetime
import traceback
from functools import wraps


class ResponseCode:
    SUCCESS = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_ERROR = 500


class StandardResponse:
    """
    标准响应
    特点：预设返回结构，包含固定的字段
    -> code，预设的状态码
    -> message，预设的信息
    -> data，返回的数据
    """

    def __init__(
            self,
            code: int = ResponseCode.SUCCESS,
            message: str = "success",
            data: Any = None,
            error: Optional[str] = None
    ):
        self.code = code
        self.message = message
        self.data = data
        self.error = error
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        response_dict = {
            "code": self.code,
            "message": self.message,
            "timestamp": self.timestamp
        }

        if self.data is not None:
            response_dict["data"] = self.data
        if self.error:
            response_dict["error"] = self.error

        return response_dict

    @staticmethod
    def success(data: Any = None, message: str = "success") -> Dict:
        """成功响应"""
        return StandardResponse(
            code=ResponseCode.SUCCESS,
            message=message,
            data=data
        ).to_dict()

    @staticmethod
    def error(
            message: str = "error",
            code: int = ResponseCode.INTERNAL_ERROR,
            error: Optional[str] = None
    ) -> Dict:
        """错误响应"""
        return StandardResponse(
            code=code,
            message=message,
            error=error
        ).to_dict()


def api_response(func):
    """统一响应装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # 如果返回的已经是StandardResponse格式，直接返回
            if isinstance(result, dict) and "code" in result:
                return jsonify(result)
            # 否则包装为成功响应
            return jsonify(StandardResponse.success(data=result))
        except Exception as e:
            # 捕获异常并返回错误响应
            error_detail = traceback.format_exc()
            return jsonify(StandardResponse.error(
                message=str(e),
                error=error_detail
            ))

    return wrapper


# 使用示例
from flask import Flask

app = Flask(__name__)


@app.route('/api/example/success')
@api_response
def example_success():
    return {
        "name": "test",
        "value": 123
    }


@app.route('/api/example/error')
@api_response
def example_error():
    raise ValueError("This is an example error")


@app.route('/api/example/custom')
def example_custom():
    return jsonify(StandardResponse.error(
        message="Custom error message",
        code=ResponseCode.BAD_REQUEST,
        error="Invalid parameter"
    ))


if __name__ == '__main__':
    app.run(debug=True)
