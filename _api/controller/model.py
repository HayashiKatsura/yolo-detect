from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, Query, HTTPException , Body

from sqlmodel import Session
from uuid import uuid4

from _api.configuration.DatabaseSession import get_session
from _api.services.YoloAPI import *
from _api.configuration.NoStandardResponse import *
from typing import List,Optional
from _api.configuration.RequestBody import *

router = APIRouter(prefix="/zjut", tags=["Model / YOLO"])

@router.post("/start-training")
async def start_training(train_params: dict):
    """
    训练前置准备 ['trainParams']
    """
    results = YoloAPI().exec_startTraining(train_params)
    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, "success", data=results).get_response_body()
    ) 

@router.delete("/stop-training/{task_id}")
async def start_training(task_id: str):
    """
    训练前置准备
    """
    results = YoloAPI().exec_stopTraining(task_id)
    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, "success", data=results).get_response_body()
    ) 


@router.get("/show-training/{task_id}")
async def show_training(
    task_id:str,
    line_no: Optional[int] = Query(None, description="起始行数，可选；若不传入则从第一行开始")
):
    """
    获取训练日志
    """
    results = YoloAPI().exec_showTraining(task_id,line_no)
    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, results['msg'], data=results).get_response_body()
    )
    
    

@router.post("/validation")
def start_validation(
    request: ValidationRequest = Body(..., description="验证参数"),
):
    """
    启动权重验证，可以一次性验证一个或多个权重文件。
    示例:
    - /weights/validate?weights_ids=1
    - /weights/validate?weights_ids=1&weights_ids=2&weights_ids=3
    """
    results = YoloAPI().exec_validation(
        request.dataset_id,
        request.conf,
        request.weights_ids
    )

    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, 'ok', data=results).get_response_body()
    )

@router.post("/prediction")
def start_prediction(
    request: PredictionRequest = Body(..., description="预测参数"),
):
    """
    启动预测
    - `weight_id`：权重文件ID
    - `files_ids`：要预测的文件ID列表
    """
    results = YoloAPI().exec_prediction(request.weight_id, request.files_ids)
    return JSONResponse(
        content=NoStandardResponse(ResponseCode.SUCCESS, 'ok', data=results).get_response_body()
    )

