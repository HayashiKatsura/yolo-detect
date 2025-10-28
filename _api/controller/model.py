from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from sqlmodel import Session
from uuid import uuid4

from _api.configuration.DatabaseSession import get_session
from _api.services.YoloAPI import *
from _api.configuration.NoStandardResponse import *
from typing import List

router = APIRouter(prefix="/zjut", tags=["Model / YOLO"])

@router.post("/start-training")
async def start_training(train_params: dict):
    """
    训练前置准备
    """
    results = YoloAPI()._pre_train(train_params)
    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, "success", data=results).get_response_body()
    ) 

@router.post("/training-log")
async def training_log(file_id:str):
    """
    获取训练日志
    """
    

@router.get("/validate")
def start_validation(
    dataset_id: int = Query(..., description="数据集ID"),
    weights_ids: List[int] = Query(..., description="要验证的权重文件ID列表，可传多个"),
):
    """
    启动权重验证，可以一次性验证一个或多个权重文件。
    示例:
    - /weights/validate?weights_ids=1
    - /weights/validate?weights_ids=1&weights_ids=2&weights_ids=3
    """
    results = YoloAPI().validation(dataset_id,weights_ids)

    return JSONResponse(content={"message": "验证完成", "results": results})

@router.post("/predict")
def start_prediction(session: Session = Depends(get_session)):
    run_id = str(uuid4())
    return {"run_id": run_id, "status": "queued"}

