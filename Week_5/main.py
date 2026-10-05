from fastapi import FastAPI

from app.database import Base, engine
from app.models.user import User
from app.routers import auth


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FastAPI Authentication API"
)

app.include_router(auth.router)


@app.get("/")
def root():
    return {
        "message": "FastAPI Auth API"
    }