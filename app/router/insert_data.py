import os
import yaml
import pandas as pd
from ..common.uri import INSERT_DATA_URI
from fastapi import APIRouter, HTTPException
from ..common.util import logger
from ..common.schema import Response, InsertData

router = APIRouter()


@router.post(INSERT_DATA_URI, response_model=Response, tags=["InsertDataset"])
def predict(input_object: InsertData):
    input_data = input_object.input_data
    if not input_data:
        logger.error("Input data is empty")
        raise HTTPException(
            status_code=406,
            detail="Input data is empty",
        )
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "training", "config.yaml")
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        path_to_data = config["extend_data"]["path_to_extend_data"]

        existing_df = pd.read_parquet(path_to_data)

        new_data_df = pd.DataFrame(input_data)
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
        combined_df.to_parquet(path_to_data, index=False)

        return Response(
            response="Data have been successfully inserted to the source data!"
        )
    except Exception as e:
        logger.error("Error occurred: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {e}"
        ) from e
