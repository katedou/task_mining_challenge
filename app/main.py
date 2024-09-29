import logging

from fastapi import APIRouter, FastAPI

from .common.config import API_VERSION, SWAGGER_TAGS, VERSION
from .common.uri import HOME_URI, SWAGGER_DOCS_URI
from .router import training, inference, delete_model, retrieve_model_info

L = logging.getLogger(__name__)

app = FastAPI(
    title="Gesture Recognition",
    version=VERSION,
    openapi_tags=SWAGGER_TAGS,
    contact={"name": "Beibei Dou", "email": "kate.dou2@gmail.com"},
    docs_url=SWAGGER_DOCS_URI,
)

# Configure router
router = APIRouter()
router.include_router(training.router)
router.include_router(inference.router)
router.include_router(retrieve_model_info.router)
router.include_router(delete_model.router)
app.include_router(router, prefix=f"/{API_VERSION}")


@app.get(HOME_URI, include_in_schema=False)
async def root():
    return {
        "message": f"Welcome to the Gesture recognition application! Use the {SWAGGER_DOCS_URI} endpoint to get started."
    }
