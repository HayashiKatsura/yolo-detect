from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any,List

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Text, BigInteger, text
from sqlalchemy.dialects.mysql import JSON as MYSQL_JSON, DATETIME as MYSQL_DATETIME

class File(SQLModel, table=True):
    __tablename__ = "files"

    # BIGINT UNSIGNED AUTO_INCREMENT
    # id: Optional[int] = Field(
    #     default=None,
    #     primary_key=True,
    #     sa_column=Column(BigInteger(unsigned=True), autoincrement=True)
    # )
    
    id: Optional[int] = Field(default=None, primary_key=True)


    # 业务类型：image/upload, image/camera, weights, dataset...
    kind: str = Field(max_length=50)

    # MIME，如 image/jpeg
    content_type: Optional[str] = Field(default=None, max_length=100)

    # 上传时文件名
    original_filename: str = Field(max_length=255)

    # 本地路径（建议相对路径）
    storage_path: str = Field(sa_column=Column(Text, nullable=False))

    # 文件大小（字节）
    # size_bytes: Optional[int] = Field(
    #     default=None,
    #     sa_column=Column(BigInteger(unsigned=True))
    # )
    size_bytes: Optional[str] = Field(default=None)


    # 结构化结果：图像/视频检测结果
    media_annotations: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(MYSQL_JSON)
    )

    # 结构化结果：权重评估指标
    model_metrics: Optional[List[dict]] = Field(default=[], sa_column=Column(MYSQL_JSON))  # 使用列表存储多个验证记录


    # 备注
    remark: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(MYSQL_JSON)
    )

    # 软删时间
    deleted_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(MYSQL_DATETIME(fsp=6), nullable=True)
    )

    # 创建/更新时间（数据库侧维护 CURRENT_TIMESTAMP(6) & ON UPDATE）
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
            "kind": self.kind,
            "content_type": self.content_type,
            "original_filename": self.original_filename,
            "storage_path": self.storage_path,
            "size_bytes": self.size_bytes,
            "media_annotations": self.media_annotations,
            "model_metrics": self.model_metrics,
            "remark": self.remark,
            "deleted_at": datetime_to_str(self.deleted_at) if self.deleted_at else None,
            "created_at": datetime_to_str(self.created_at),
            "updated_at": datetime_to_str(self.updated_at)
        }
