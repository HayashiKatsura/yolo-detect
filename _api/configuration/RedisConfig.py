import redis
from typing import Optional


_REDIS_CONFIG = {
    "account": "root",
    'password':'y4hcXLDwkpepZWnbSDpRgMxagbhB8',
    'host':'8.153.64.199',
    'port': 6379,
    'db':0
}



class RedisConfig:
    def __init__(self, 
                 host: str = _REDIS_CONFIG['host'], 
                 port: int = _REDIS_CONFIG['port'], 
                 db: int = _REDIS_CONFIG['db'], 
                 password: Optional[str] = _REDIS_CONFIG['password']):
        self.host = host
        self.port = port
        self.db = db
        self.password = password  # Redis 密码
        self.redis_client: Optional[redis.StrictRedis] = None

    def connect(self):
        """ 连接 Redis，支持密码验证 """
        if self.password:
            self.redis_client = redis.StrictRedis(host=self.host, port=self.port, db=self.db, password=self.password, decode_responses=True)
        else:
            self.redis_client = redis.StrictRedis(host=self.host, port=self.port, db=self.db, decode_responses=True)
        
        # 可选：验证连接
        if self.redis_client.ping():
            print(f"Redis connected successfully to {self.host}:{self.port}")
        else:
            raise ConnectionError(f"Could not connect to Redis at {self.host}:{self.port}")

    def get_client(self) -> redis.StrictRedis:
        """ 获取 Redis 客户端 """
        if self.redis_client is None:
            self.connect()
        return self.redis_client

# 创建 Redis 配置实例
redis_config = RedisConfig()