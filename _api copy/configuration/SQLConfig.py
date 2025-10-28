DATABAE_ACCOUNT_PASSWORD = {
    "account": "root",
    # "password": "123456",
    # 'host': '127.0.0.1',
    'password':'y4hcXLDwkpepZWnbSDpRgMxagbhB8',
    'host':'8.153.64.199',
    'port': 3306,
    'database':'anomalies_detect'
}

class SQLConfig:
    SQLALCHEMY_DATABASE_URI = \
            f"mysql+pymysql://{DATABAE_ACCOUNT_PASSWORD['account']}:" \
            f"{DATABAE_ACCOUNT_PASSWORD['password']}@" \
            f"{DATABAE_ACCOUNT_PASSWORD['host']}:" \
            f"{DATABAE_ACCOUNT_PASSWORD['port']}/" \
            f"{DATABAE_ACCOUNT_PASSWORD['database']}?charset=utf8mb4"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 1800
    }

class DevelopmentConfig(SQLConfig):
    DEBUG = True

class ProductionConfig(SQLConfig):
    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://prod_user:prod_pass@dbserver:3306/prod_db?charset=utf8mb4"
