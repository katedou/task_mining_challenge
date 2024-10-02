import os
import json
import yaml
from ..common.uri import RETRIEVE_MODEL_URI
from fastapi import APIRouter, HTTPException
from ..common.util import logger
from ..common.schema import RetrieveModelResult, ModelVersion

router = APIRouter()


@router.post(
    RETRIEVE_MODEL_URI,
    response_model=RetrieveModelResult,
    tags=["RetrieveModelInformation"],
)
def retrieve_model(input_object: ModelVersion):
    model_version = input_object.model_version
    if not model_version:
        logger.error("Input model version is empty")
        raise HTTPException(
            status_code=406,
            detail="Input model version is empty",
        )
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "training", "config.yaml")
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        path_to_model_metrics = os.path.join(
            config["train"]["model_root_path"],
            str(model_version),
            "best_model_metrics.json",
        )

        with open(path_to_model_metrics, "r") as metrics_file:
            model_metrics = json.load(metrics_file)

        return RetrieveModelResult(
            model_version=str(model_version),
            accuracy=model_metrics["accuracy"],
            eval_loss=model_metrics["eval_loss"],
            hyperparameter=model_metrics["hyperparameters"],
        )
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        ) from e
