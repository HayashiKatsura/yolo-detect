from datetime import datetime

# from sympy import pprint


class NoStandardResponse:
    """
    非标准响应
    特点：实时定义返回结构，而非预设的返回结构
    -> 只包含必要的字段，时间戳
    -> code， 且code 为非预设的状态码，想要什么状态码就返回什么状态码
    -> msg， 且msg 为非预设的信息，想要什么信息就返回什么信息
    """

    def __init__(self, code: int, msg: str, **kwargs):
        self.code = code
        self.msg = msg
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.kwargs = kwargs

    def get_response_body(self):

        response_body = {
            "code": self.code,
            "msg": self.msg,
            "timestamp": self.timestamp
        }

        for key, value in self.kwargs.items():
            if key not in response_body:
                response_body[key] = value


        return response_body


if __name__ == '__main__':
    print(NoStandardResponse(200, "success", data={"name": "xiaoming"}).get_response_body())
