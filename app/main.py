from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Drone Eye API")

app.include_router(router)
