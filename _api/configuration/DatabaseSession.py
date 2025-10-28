from sqlmodel import SQLModel, create_engine, Session
from typing import Generator
from .SQLConfig import sqlalchemy_url

engine = create_engine(sqlalchemy_url(), echo=False)

# 在 main.py 的启动事件或独立脚本里执行： SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session