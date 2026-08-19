"""FastAPI entrypoint. Uvicorn loads `app` from this module."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, request_id_var
from app.runtime import build_container
from app.api.routes.documents import router as documents_router, sample_router
from app.api.routes.health import router as health_router
from app.api.routes.qa import router as qa_router


def create_app(
    settings: Settings | None = None,
    llm=None,
    embeddings=None,
) -> FastAPI:
    """Build the HTTP app. Tests inject a stub LLM and fake embeddings here."""
    configure_logging()
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.container = build_container(settings, llm=llm, embeddings=embeddings)
        yield

    application = FastAPI(
        title="Document Intelligence Agent",
        description="Document-grounded question answering over PDF and JSON files.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = request_id
        return response

    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        request_id = request.headers.get("x-request-id") or request_id_var.get()
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "code": exc.code, "request_id": request_id},
        )

    application.include_router(health_router)
    application.include_router(documents_router)
    application.include_router(sample_router)
    application.include_router(qa_router)
    return application


app = create_app()
