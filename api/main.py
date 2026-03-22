from fastapi import FastAPI

from api.routes.simulation import router as simulation_router
from api.routes.data import router as data_router
from db.database import init_db

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(
    title="F1 Strategy Intelligence API",
    version="1.0.0",
)

app.include_router(simulation_router)
app.include_router(data_router)


@app.on_event("startup")
def startup_event():
    """Create database tables if they don't exist (needed for Render deploy)."""
    init_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}
