# Task Mining Challenge

## Run the application locally 

**Pre-requirements**: 
- Python 3.11
- Task installed on your machine. For installation details, see [here](https://taskfile.dev/installation/)
- Docker installed on your machine if you want to run the application locally with Docker
- To use the Weights & Biases Board, you need to export the `WANDB_API_KEY` locally. Run `export WANDB_API_KEY="your api key"`.

### Run the Gesture Recognition application locally

1. Create your local virtual environment by running:

```bash
task create-venv
task install-dependencies
```
2. Activate your virtual environment by running `source ./venv/bin/activate`
3. Run the application locally:
```bash
export WANDB_API_KEY="your api key"
task run-local
```
Then open your browser and go to `http://localhost:5000/docs`. You will be directed to the SwaggerUI, where you can trigger the API endpoints.

### Run the Gesture Recognition application with Docker Container

To run this application with a Docker container locally, simply run the following lines:
```bash
# build your docker container locally 
export WANDB_API_KEY="your api key" 
task docker-build
# run the built docker container
task docker-run
```
Then you can open the localhost by copying `http://localhost:5000/docs` to your browser. You can then try out the different API calls in the SwaggerUI.

### Test the API calls

To test the '/insert_data' endpoint, you can find a sample input in `test/insert_data.json`.
To test the '/inference' endpoint, you can find a sample input for the input_data part in `test/inference_input.json`.

## Run Training locally 

For developers and data scientists who want to run the training locally with fine-tuning of hyperparameters, changing optimizers, applying other technologies, or changing model architecture, you can use the following instructions.

### Trigger training locally
- Activate the virtual environment by running the following command:

    ```bash 
    source .venv/bin/activate
    ```

    ```bash
    python -m training.train --cfg ./training/config.yaml --model_version 2
    ```

### Trigger pytest
1. Activate the virtual environment `source ./venv/bin/activate`.
2. Run `task pytest-run`.


### Todos for making this application production-ready:

- To better manage and scale this project, it is suggested to deploy the whole solution to the cloud. For example, the entire solution can be moved to Azure cloud, where Azure Blob Storage can be used for storing models and data, Azure Container Registry for managing container images, and Azure Container/Kubernetes Service for better scaling of the current solution, etc.