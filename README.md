# Task Mining Challenge

## Run the application locally 
Pre-requirement: 
- You have Task installed on your machine. More details about how to install it, you can dinit in [here](https://taskfile.dev/installation/)
- You have docker on your machine if you want to run the application locally with docker.
- To be able to use the Weights&Biases Board, you need to export the `WANDB_API_KEY` locally. Then run `ENV WANDB_API_KEY=$WANDB_API_KEY`.


### Run the Gesture recognition application locally

1. Create your local virtual environment by running 
```bash
task create-venv
task install-dependencies
```
2. Activate your virtual environment by running `source ./venv/bin/activate`
3. Run the application locally:
```bash
task run-local
```
Then open your `localhost` in browser, go to `http://localhost:5000/docs`. The you will directed to the SwaggerUI, where you can trigger the api endpionts.

### Run the Gesture Recognition application with Docker Container


### Todo:

To better manage and scale this project, it is suggest to deploy the whole solution to cloud.