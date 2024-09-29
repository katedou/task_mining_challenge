FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

COPY . /app

ARG WANDB_API_KEY
# this is needed for the weights and bias
ENV WANDB_API_KEY=$WANDB_API_KEY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r /app/requirements.txt


EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000", "--timeout-keep-alive", "15"]