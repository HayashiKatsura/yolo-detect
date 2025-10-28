from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Text, BigInteger, text
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME


class Dataset(SQLModel, table=True):
    __tablename__ = "datasets"

    # id: Optional[int] = Field(
    #     default=None,
    #     primary_key=True,
    #     sa_column=Column(BigInteger(unsigned=True), autoincrement=True)
    # )
    id: Optional[int] = Field(default=None, primary_key=True)


    name: str = Field(max_length=255)

    # 数据集类型：yolo/coco/voc/custom ...
    kind: str = Field(default="yolo", max_length=50)

    # 根目录（建议相对路径）
    root_path: str = Field(sa_column=Column(Text, nullable=False))

    # 配置文件路径（相对 root_path 或绝对）
    yaml_path: Optional[str] = Field(default=None, sa_column=Column(Text))

    # 图像/标签目录名（相对 root_path）
    images_dir: Optional[str] = Field(default=None, max_length=255)
    labels_dir: Optional[str] = Field(default=None, max_length=255)

    # 计数
    train_counts: int = Field(default=0)
    last_train_counts: int = Field(default=0)
    val_counts: int = Field(default=0)
    last_val_counts: int = Field(default=0)
    test_counts: int = Field(default=0)
    last_test_counts: int = Field(default=0)

    # 数据集总大小（字节）
    # size_bytes: Optional[int] = Field(
    #     default=None,
    #     sa_column=Column(BigInteger(unsigned=True))
    # )
    size_bytes: Optional[int] = Field(default=None)


    # 备注
    remark: Optional[str] = Field(default=None, sa_column=Column(Text))

    # 软删时间
    deleted_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(MYSQL_DATETIME(fsp=6), nullable=True)
    )

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
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "root_path": self.root_path,
            "yaml_path": self.yaml_path,
            "images_dir": self.images_dir,
            "labels_dir": self.labels_dir,
            "train_counts": self.train_counts,
            "last_train_counts": self.last_train_count
        }
