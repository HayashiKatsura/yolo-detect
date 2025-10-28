from pydantic_settings import BaseSettings

_DATABAE_CONFIG = {
    "sql": "mysql",
    "account": "root",
    'password':'y4hcXLDwkpepZWnbSDpRgMxagbhB8',
    'host':'8.153.64.199',
    'port': 3306,
    'database':'chips_yolo'
}

class SQLConfigSettings(BaseSettings):
    DB_DRIVER: str = _DATABAE_CONFIG["sql"]              
    DB_HOST: str = _DATABAE_CONFIG["host"]
    DB_PORT: int = _DATABAE_CONFIG["port"]                        
    DB_USER: str = _DATABAE_CONFIG["account"]
    DB_PASSWORD: str = _DATABAE_CONFIG["password"]
    DB_NAME: str = _DATABAE_CONFIG["database"]

    # # Celery / MQ
    # RABBITMQ_URL: str = "amqp://guest:guest@127.0.0.1:5672//"
    # CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/0"  # 可选

    # class Config:
    #     env_file = ".env"

settings = SQLConfigSettings()

def sqlalchemy_url() -> str:
    if settings.DB_DRIVER == "sqlite":
        return f"sqlite:///./{settings.DB_NAME}.db"
    return f"{settings.DB_DRIVER}+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
