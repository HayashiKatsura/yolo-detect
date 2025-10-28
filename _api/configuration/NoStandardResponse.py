from datetime import datetime
from typing import Any, Dict
from enum import Enum


class ResponseCode(Enum):
    """统一响应码定义"""
    SUCCESS = (200, "成功")
    EMPTY_DATA = (255, "空数据")
    NOT_FOUND = (404, "找不到文件")
    UNKNOWN_ERROR = (500, "未知错误")

    def __init__(self, code: int, message: str):
        self._code = code
        self._message = message

    @property
    def code(self) -> int:
        return self._code

    @property
    def message(self) -> str:
        return self._message

    def __str__(self):
        return f"{self._code} {self._message}"


class NoStandardResponse:
    """
    自定义统一响应结构
    - code: 响应码（来自 ResponseCode）
    - msg: 响应信息（自定义或默认）
    - timestamp: 响应时间
    - kwargs: 额外字段（如 data）
    """

    def __init__(self, code: ResponseCode, msg: str = None, **kwargs: Any):
        self.code = code.code
        self.msg = msg or code.message  # 如果没传 msg，用枚举默认消息
        self.timestamp = datetime.now().isoformat(timespec="seconds")
        self.kwargs = kwargs

    def get_response_body(self) -> Dict[str, Any]:
        body = {
            "code": self.code,
            "msg": self.msg,
            "timestamp": self.timestamp
        }
        body.update(self.kwargs)
        return body


# ✅ 示例演示
if __name__ == "__main__":
    # 自定义消息
    r1 = NoStandardResponse(ResponseCode.SUCCESS, "操作成功", data={"user": "xiaoming"})
    print(r1.get_response_body())

    # 使用默认消息（不传 msg）
    r2 = NoStandardResponse(ResponseCode.NOT_FOUND)
    print(r2.get_response_body())
    # JSONResponse(content=success(data=result))
