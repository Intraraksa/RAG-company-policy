import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import health, query
from app.core.config import settings
from app.services.errors import ServiceError

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="RAG AWS Service")


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


app.include_router(health.router)
app.include_router(query.router)
