from sqlalchemy import create_engine, text

# 配置数据库连接
DATABASE_URI = "mysql+pymysql://root:123456@localhost:3306/anomalies_detect?charset=utf8mb4"

# 创建数据库连接引擎
engine = create_engine(DATABASE_URI)

# 执行查询操作并返回字典格式的结果
def test_sql_query():
    with engine.connect() as connection:
        # 执行 SQL 查询
        result = connection.execute(text("SELECT file_id, file_name FROM files"))
        
        # 获取字典格式的结果
        rows = result.mappings().all()  # 返回所有结果，字典格式
        
        print('rows:',rows)
        # 打印查询结果
        for row in rows:
            print(f"File ID: {row['file_id']}, File Name: {row['file_name']}")

if __name__ == "__main__":
    test_sql_query()  # 调用函数进行查询