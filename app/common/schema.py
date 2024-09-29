from pydantic import BaseModel
from typing import Dict

from .config import VERSION


class ModelVersion(BaseModel):
    model_version: int


class Response(BaseModel):
    response: str
    version: str = VERSION


class InferenceInput(BaseModel):
    model_version: str
    input_data: Dict[str, float]


class InferenceResult(BaseModel):
    prediction: str
    model_version: str
    probability: float


class RetrieveModelResult(BaseModel):
    accuracy: float
    model_version: str
    eval_loss: float
    hyperparameter: dict
