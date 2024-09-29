import argparse
import json
import os

import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from training.dataloader import DataPreprocessor, GestureData
from training.model import ShallowNet


def main(path_to_config_file: str, model_version: int):
    """
    trigger the training process of the Shallow Net
    """
    wandb.init(project="gesture_classification", name="shallow_nn_experiment")

    cfg = yaml.load(open(path_to_config_file, "r"), Loader=yaml.FullLoader)
    train_config = cfg["train"]

    # load in data with preprocessing
    processed_data = DataPreprocessor(model_version, cfg)

    train_dataset = GestureData(processed_data.X_train_scaled, processed_data.y_train)
    test_dataset = GestureData(processed_data.X_test_scaled, processed_data.y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=train_config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config["batch_size"], shuffle=False
    )

    # Simplified hyperparameters handling
    hyperparameters = {
        "input_dim": train_config["input_dim"],
        "hidden_dims": train_config["hidden_dims"],
        "output_dim": train_config["output_dim"],
        "momentum": train_config["momentum"],
        "learning_rate": train_config["learning_rate"],
        "num_epochs": train_config["num_epochs"],
        "batch_size": train_config["batch_size"],
    }

    model = ShallowNet(
        hyperparameters["input_dim"],
        hyperparameters["hidden_dims"],
        hyperparameters["output_dim"],
    )
    # define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        momentum=hyperparameters["momentum"],
    )

    # Log model architecture and hyperparameters
    wandb.watch(model, log="all")
    wandb.config.update(hyperparameters)

    # Initialize variables to keep track of best performance
    best_accuracy = 0.0
    model_base_output = os.path.join(
        train_config["model_root_path"], str(model_version)
    )
    os.makedirs(model_base_output, exist_ok=True)
    best_model_path = os.path.join(model_base_output, "best_model.pth")
    best_metrics_path = os.path.join(model_base_output, "best_model_metrics.json")

    # start the training
    for epoch in range(hyperparameters["num_epochs"]):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Log training loss every 100 batches
            if i % 100 == 99:
                wandb.log(
                    {
                        "batch": epoch * len(train_loader) + i,
                        "train_loss": running_loss / 100,
                    }
                )
                running_loss = 0.0

        # Evaluation
        model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                eval_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average evaluation loss and accuracy
        avg_eval_loss = eval_loss / len(test_loader)
        accuracy = 100 * correct / total

        # Log evaluation metrics
        wandb.log(
            {
                "epoch": epoch,
                "eval_loss": avg_eval_loss,
                "accuracy": accuracy,
                "hyperparameters": hyperparameters,
            }
        )

        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            # Save the model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

            # Save the metrics
            metrics = {
                "accuracy": accuracy,
                "eval_loss": avg_eval_loss,
                "epoch": epoch,
                "hyperparameters": hyperparameters,
            }
            with open(best_metrics_path, "w") as f:
                json.dump(metrics, f)

            # Log best model to wandb
            wandb.run.summary["best_accuracy"] = best_accuracy
            wandb.save(best_model_path)

        # Print epoch statistics
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{hyperparameters['num_epochs']}], Train Loss: {running_loss/len(train_loader):.4f}, "
                f"Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.2f}%"
            )

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to the config file")
    parser.add_argument(
        "--model_version", type=int, help="The model version you going to train"
    )
    args = parser.parse_args()
    cfg_file = args.cfg
    model_version = args.model_version
    main(cfg_file, model_version=model_version)
