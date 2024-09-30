# Task Mining Challenge

## Run the application locally 

**Pre-requirements**: 
- Python 3.11
- Task installed on your machine. For installation details, see [here](https://taskfile.dev/installation/)
- Docker installed on your machine if you want to run the application locally with Docker
- To use the Weights & Biases Board, you need to export your `WANDB_API_KEY` locally. Run `export WANDB_API_KEY="your api key"`. If you have no Weights & Biases account and do not want to create one, I can provide you a key that is ready for sharing on the interview day.
- For the following run the gesture recognition locally, the trigger training locally and trigger pytest locally, you shall following the instruction under the home directory of this project, i.e.`/your_directory/task_mining_challenge`.

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
Then open your browser and go to `http://localhost:5000/docs`. You will be directed to the SwaggerUI, where you can trigger the API endpoints. The **User name and password** to access the swaggerUI is defined in `./app/main.py`, which are only the dummy user name and password, just for the sake of showing the authorization functionaity. **Note:** Saving authorization token in the script is not proper approach in practice. As here is the only the dummy pass and user name, and only for showcasing the functionality, so they are included in the source code. In practice, consider use Key Vault or Database for saving this information.

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

- To test the `/insert_data` endpoint, you can find a sample input in `test/insert_data.json`.
- To test the `/inference` endpoint, you can find a sample input for the **input_data** part in `test/inference_input.json`, you still need to provide the model version to use for the inference. It is suggested to use model version **1**, which was trained with 800 epochs and the best model is saved during the training.
- To test `/training` endpoint, by default in the `./trainning/config.yaml` the number of epochs is defined as 100. If you just want to test this oout and make it run faster, you can set this parameter to for example: 10. The trained model and the used MIC/Standard Scaler can be found in `./training/models/` and `./training/data/preprocess_xx_data/`.
- To test `/delete_model` endpoint, you will find that the model and it's MIC/Standard Scaler will be deleted. 

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

## Trigger pytest locally
1. Activate the virtual environment `source ./venv/bin/activate`.
2. Run `task pytest-run`.


## Todos for making this application production-ready:
1. Cloud deployment: Move the solution to a cloud platform (e.g., Azure) for better management and scalability.
   - Use Azure Blob Storage for model and data storage
   - Utilize Azure Container Registry for container image management
   - Deploy to Azure Container/Kubernetes Service for improved scaling

2. Implement data versioning to track changes in datasets over time.

3. Adapt model training tracking:
   - Currently using Weights & Biases
   - Consider enterprise-specific limitations or preferred toolsets
   - Integrate with the company's existing infrastructure

4. Code refactoring:
   - Add type hints for function inputs and outputs
   - Improve documentation for all functions
   - Refactor some scripts to make them more well structured. For exampl: `./training/train.py`

5. Enhance testing:
   - Add comprehensive unit tests
   - Expand integration test coverage

6. Separate model training and inference:
   - Split into distinct services or platforms
   - Optimize resource allocation for training
   - Ensure consistent inference performance
   - Improve model tracking and versioning

7. Implement CI/CD pipeline:
   - Automate orchestration
   - Include automated testing in the pipeline
   - Streamline cloud deployment
   - Facilitate smooth model releases

8. Security enhancements:
   - Implement proper authentication and authorization
   - Secure API endpoints
   - Manage secrets and credentials securely

9. Monitoring and logging(Toolset: Azure Log Analytics/Datadog):
   - Set up comprehensive logging
   - Implement performance monitoring
   - Create alerts for critical issues

10. Scalability and load balancing:
    - Design for horizontal scaling
    - Implement load balancing for high availability

11. Error handling and resilience:
    - Improve error handling throughout the application
    - Implement retry mechanisms for transient failures
