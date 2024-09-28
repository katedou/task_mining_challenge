import argparse
import os

import joblib
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from training.dataloader import GestureData
from training.model import ShallowNet


def evaluation(config_file_path: str):
    cfg = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
    train_config = cfg["train"]
    test_config = cfg["test"]

    model = ShallowNet(
        input_dim=train_config["input_dim"],
        hidden_dims=train_config["hidden_dims"],
        output_dim=train_config["output_dim"],
    )
    state_dict = torch.load(test_config["path_to_model"], weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    total = 0

    # load hyper parameters from config file
    output_dim = train_config["output_dim"]

    # Load test data
    # Assuming you've saved and loaded your test data as discussed earlier
    X_test = joblib.load(os.path.join(test_config["path_to_test_data"], "X.joblib"))
    y_test = joblib.load(os.path.join(test_config["path_to_test_data"], "y.joblib"))

    # Convert to numpy arrays
    if isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # Convert to PyTorch tensors
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    test_dataset = GestureData(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=train_config["batch_size"], shuffle=False
    )

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

    # Print some additional information for debugging
    print(f"Number of classes: {output_dim}")


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file")
    args = parser.parse_args()
    cfg_file = args.cfg
    evaluation(cfg_file)
