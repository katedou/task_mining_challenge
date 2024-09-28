import logging

# API
API_VERSION = "v0"

# Logging
LOGGING_LEVEL = logging.INFO


# Swagger
SWAGGER_TAGS = [
    {
        "name": "Training",
        "description": "Trigger training of the model using existing data and stores the new model alongside its evaluation metrics.",
    },
    {
        "name": "Inference",
        "description": "Perform inference on new data points. Users can use older versions of the model.",
    },
    {
        "name": "Retrive Model Information",
        "description": "Retrieve model versions evaluation metrics.",
    },
    {
        "name": "Delete model",
        "description": "Delete specific model version.",
    },
    {
        "name": "Extend existing dataset",
        "description": "Extend the dataset with new data",
    },
]

VERSION = "1.0.0"