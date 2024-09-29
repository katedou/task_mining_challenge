import torch
import json
import argparse
import os
import pandas as pd
import yaml
from training.model import ShallowNet
from training.dataloader import DataPreprocessor
import torch.nn.functional as F


def make_inference(model_version, input_data, config_file_path):
    """
    make inference based on the given version of model
    """

    cfg = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
    # prepare data for infrencing
    df = pd.DataFrame([input_data])
    data_processor = DataPreprocessor(train=False).load_for_inference(
        model_version=model_version, config=cfg
    )
    preprocessed_data = data_processor.prepare_for_inference(df)

    # Load the model
    model = ShallowNet(
        input_dim=cfg["train"]["input_dim"],
        hidden_dims=cfg["train"]["hidden_dims"],
        output_dim=cfg["train"]["output_dim"],
    )

    model_path = os.path.join(
        cfg["test"]["base_path_to_model"], str(model_version), "best_model.pth"
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Convert preprocessed data to PyTorch tensor
    input_tensor = torch.from_numpy(preprocessed_data).float()

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        # Process output (e.g., get predicted class)
        prob, predicted = torch.max(probabilities, 1)

    return predicted.item() + 1, prob.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file")
    parser.add_argument(
        "--model_version", type=int, help="The model version you going to train"
    )
    parser.add_argument(
        "--json_input", type=str, help="The model version you going to train"
    )
    args = parser.parse_args()
    cfg_file = args.cfg
    model_version = args.model_version
    with open(args.json_input, "r") as f:
        test_input = json.load(f)
    input_data = test_input["input_data"]
    prediction, probability = make_inference(model_version, input_data, cfg_file)
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability}")
