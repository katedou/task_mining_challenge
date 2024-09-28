from pydantic import BaseModel

from .config import VERSION


class ModelVersion(BaseModel):
    model_version: int


class StringList(BaseModel):
    texts: list[str]


class Training(BaseModel):
    response: str
    version: str = VERSION
