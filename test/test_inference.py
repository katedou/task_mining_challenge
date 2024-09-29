import pytest
import asyncio
from training.inference import make_inference
import os
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def config_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "training", "config.yaml")


@pytest.fixture
def model_version():
    return "1"


def sample_input():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(current_dir, "inference_input.json")
    with open(input_file_path, "r") as f:
        data = json.load(f)
    return data["input_data"]


@pytest.mark.asyncio
async def test_concurrent_inference(config_path, model_version):
    input_data = sample_input()

    # Create 5 concurrent inference tasks
    async def single_inference():
        prediction, probability = await make_inference(
            model_version, input_data, config_path
        )
        assert isinstance(prediction, int)
        assert 0 <= probability <= 1
        return prediction, probability

    tasks = [single_inference() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Check that we got 5 results
    assert len(results) == 5

    # Check that each result is valid
    for prediction, probability in results:
        assert isinstance(prediction, int)
        assert 0 <= probability <= 1


@pytest.mark.asyncio
async def test_inference_performance(config_path, model_version):
    input_data = sample_input()
    start_time = asyncio.get_event_loop().time()

    async def single_inference():
        return await make_inference(model_version, input_data, config_path)

    tasks = [single_inference() for _ in range(5)]
    await asyncio.gather(*tasks)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    logging.debug(
        f"Total running time for concurrent 5 inference calls is {total_time}s"
    )
    # Assert that all 5 inferences completed within a reasonable time (e.g., 5 seconds)
    assert total_time < 5, f"Inference took too long: {total_time} seconds"
