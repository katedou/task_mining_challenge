import os
from ..common.uri import INFERECE_URI
from fastapi import APIRouter, HTTPException
from ..common.util import logger
from ..common.schema import InferenceInput, InferenceResult
from training.inference import make_inference

router = APIRouter()


@router.post(INFERECE_URI, response_model=InferenceResult, tags=["Inference"])
def predict(input_object: InferenceInput):
    model_version = input_object.model_version
    input_data = input_object.input_data
    if not model_version:
        logger.error("Input model version is empty")
        raise HTTPException(
            status_code=406,
            detail="Input model version is empty",
        )
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "training", "config.yaml")

        prediction, probability = make_inference(model_version, input_data, config_path)
        return InferenceResult(
            predicted_class=str(prediction),
            probability=probability,
            model_version=model_version,
        )
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        ) from e
