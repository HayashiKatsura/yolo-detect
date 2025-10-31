from typing import List, Union

from pydantic import BaseModel

class DeleteFilesRequest(BaseModel):
    file_ids: List[Union[str, int]]

class PredictionRequest(BaseModel):
    weight_id: int
    files_ids: List[int]
    
class ValidationRequest(BaseModel):
    dataset_id: int
    conf : float 
    weights_ids: List[int]