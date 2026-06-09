from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import init_db
from app.routers import upload, query

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Startup complete.")
    yield

app = FastAPI(title="Enterprise RAG", lifespan=lifespan)

app.include_router(upload.router)
app.include_router(query.router)

@app.get("/health")
def health():
    return {"status": "ok"}