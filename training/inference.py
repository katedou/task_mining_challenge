import torch
import json
import argparse
import os
import pandas as pd
import yaml
from training.model import ShallowNet
from training.dataloader import DataPreprocessor
import torch.nn.functional as F
import asyncio

# Global variables to store loaded models and data processors
model_cache = {}
data_processor_cache = {}


async def load_model_and_processor(model_version, config_file_path):
    """
    Asynchronously load the model and data processor
    """
    if model_version in model_cache:
        return model_cache[model_version], data_processor_cache[model_version]

    cfg = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)

    # Load data processor
    data_processor = DataPreprocessor(train=False).load_for_inference(
        model_version=model_version, config=cfg
    )

    # Load the model
    model = ShallowNet(
        input_dim=cfg["train"]["input_dim"],
        hidden_dims=cfg["train"]["hidden_dims"],
        output_dim=cfg["train"]["output_dim"],
    )

    model_path = os.path.join(
        cfg["test"]["base_path_to_model"], str(model_version), "best_model.pth"
    )

    state_dict = await asyncio.to_thread(
        torch.load, model_path, map_location=torch.device("cpu"), weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Cache the model and data processor
    model_cache[model_version] = model
    data_processor_cache[model_version] = data_processor

    return model, data_processor


async def make_inference(model_version, input_data, config_file_path):
    """
    Make inference based on the given version of model
    """
    model, data_processor = await load_model_and_processor(
        model_version, config_file_path
    )

    # Prepare data for inferencing
    df = pd.DataFrame([input_data])
    preprocessed_data = await asyncio.to_thread(
        data_processor.prepare_for_inference, df
    )

    # Convert preprocessed data to PyTorch tensor
    input_tensor = torch.from_numpy(preprocessed_data).float()

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prob, predicted = torch.max(probabilities, 1)

    # Plus 1 to the prediction, as PyTorch counts labels start from 0
    # We need to shift our label to its original version
    return predicted.item() + 1, prob.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file")
    parser.add_argument(
        "--model_version", type=int, help="The model version you're going to use"
    )
    parser.add_argument("--json_input", type=str, help="Path to the JSON input file")
    args = parser.parse_args()
    cfg_file = args.cfg
    model_version = args.model_version
    with open(args.json_input, "r") as f:
        test_input = json.load(f)
    input_data = test_input["input_data"]

    async def run_inference():
        prediction, probability = await make_inference(
            model_version, input_data, cfg_file
        )
        print(f"Prediction: {prediction}")
        print(f"Probability: {probability}")

    asyncio.run(run_inference())
