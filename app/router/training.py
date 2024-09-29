import os
from fastapi import APIRouter, HTTPException

from training.train import main

from ..common.schema import ModelVersion, Response
from ..common.uri import TRAINING_URI
from ..common.util import logger


router = APIRouter()


@router.post(TRAINING_URI, response_model=Response, tags=["Training"])
def trigger_training(input_object: ModelVersion):
    model_version = input_object.model_version
    if not model_version:
        logger.error("Input model version is empty")
        raise HTTPException(
            status_code=406,
            detail="Input model version is empty",
        )
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the config file
        config_path = os.path.join(current_dir, "..", "..", "training", "config.yaml")
        main(
            path_to_config_file=config_path,
            model_version=model_version,
        )
        return Response(response="Training had been successfully executed!")
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        ) from e
