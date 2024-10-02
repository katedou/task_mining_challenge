import os
import shutil
import yaml
from ..common.uri import DELETE_MDOEL_URI
from fastapi import APIRouter, HTTPException
from ..common.util import logger
from ..common.schema import Response, ModelVersion

router = APIRouter()


@router.delete(DELETE_MDOEL_URI, response_model=Response, tags=["DeleteModel"])
def delete_model(input_object: ModelVersion):
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

        paths_to_delete = [
            os.path.join(config["train"]["model_root_path"], str(model_version)),
            os.path.join(config["train"]["train_data_root_path"], str(model_version)),
            os.path.join(config["train"]["test_data_root_path"], str(model_version)),
        ]

        for path in paths_to_delete:
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info(f"Deleted directory: {path}")
            else:
                logger.warning(f"Directory not found: {path}")

        return Response(response=f"Successfully deleted model version {model_version}")
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        ) from e
