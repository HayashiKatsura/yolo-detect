from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Text, BigInteger, text
from sqlalchemy.dialects.mysql import JSON as MYSQL_JSON, DATETIME as MYSQL_DATETIME


class DatasetRun(SQLModel, table=True):
    __tablename__ = "dataset_runs"

    # id: Optional[int] = Field(
    #     default=None,
    #     primary_key=True,
    #     sa_column=Column(BigInteger(unsigned=True), autoincrement=True)
    # )
    id: Optional[int] = Field(default=None, primary_key=True)


    # 逻辑外键：指向 datasets.id（不加真实 FK）
    dataset_id: int = Field(default=None, max_length=36)


    # 可选对外ID
    run_uuid: Optional[str] = Field(default=None, max_length=36)

    # 逻辑外键：指向 files.id（训练使用或产出的权重）
    model_file_id: Optional[int] = Field(default=None)

    # 任务状态：queued/running/succeeded/failed
    status: str = Field(default="queued", max_length=32)

    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(MYSQL_DATETIME(fsp=6))
    )
    finished_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(MYSQL_DATETIME(fsp=6))
    )

    # 指标 / 产物 / 超参
    metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(MYSQL_JSON))
    artifacts: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(MYSQL_JSON))
    params: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(MYSQL_JSON))

    # 备注
    remark: Optional[str] = Field(default=None, sa_column=Column(Text))

    # 创建/更新时间
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            MYSQL_DATETIME(fsp=6),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP(6)")
        )
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            MYSQL_DATETIME(fsp=6),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP(6)"),
            onupdate=text("CURRENT_TIMESTAMP(6)")
        )
    )
    
    def to_dict(self):
        def datetime_to_str(dt):
            if isinstance(dt, datetime):
                return dt.strftime("%Y-%m-%d %H:%M:%S")  
            return dt 
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "run_uuid": self.run_uuid,
            "model_file_id": self.model_file_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "params": self.params,
            "remark": self.remark,
            "created_at": datetime_to_str(self.created_at),
            "updated_at": datetime_to_str(self.updated_at)
        }
