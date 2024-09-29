import logging
from typing import Dict
import secrets
from fastapi import APIRouter, FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from .common.config import API_VERSION, SWAGGER_TAGS, VERSION
from .common.uri import HOME_URI, SWAGGER_DOCS_URI
from .router import training, inference, delete_model, retrieve_model_info, insert_data

L = logging.getLogger(__name__)


security = HTTPBasic()

# reference for the simple authorization : https://medium.com/@abdulwasa.abdulkader/how-to-implement-a-simple-role-based-access-control-rbac-in-fastapi-using-middleware-af07d31efa9f
# Simulated user database (replace with a real database in production)
USERS: Dict[str, Dict[str, str]] = {
    "admin": {"username": "admin", "password": "adminpass", "role": "admin"},
    "user": {"username": "user", "password": "userpass", "role": "user"},
}


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = USERS.get(credentials.username)
    if user is None or not secrets.compare_digest(
        credentials.password, user["password"]
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user


def admin_only(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


app = FastAPI(
    title="Gesture Recognition",
    version=VERSION,
    openapi_tags=SWAGGER_TAGS,
    contact={"name": "Beibei Dou", "email": "kate.dou2@gmail.com"},
    docs_url=SWAGGER_DOCS_URI,
)

# Configure router
router = APIRouter()
router.include_router(training.router, dependencies=[Depends(get_current_user)])
router.include_router(inference.router, dependencies=[Depends(get_current_user)])
router.include_router(
    retrieve_model_info.router, dependencies=[Depends(get_current_user)]
)
router.include_router(
    delete_model.router, dependencies=[Depends(admin_only)]
)  # Admin only
router.include_router(insert_data.router, dependencies=[Depends(admin_only)])
app.include_router(router, prefix=f"/{API_VERSION}")


@app.get(HOME_URI, include_in_schema=False)
async def root(user: dict = Depends(get_current_user)):
    return {
        "message": f"Welcome to the Gesture recognition application, {user['username']}! Use the {SWAGGER_DOCS_URI} endpoint to get started."
    }
