"""FastAPI dependency that exposes the process-wide AppContainer."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.runtime import AppContainer


def get_container(request: Request) -> AppContainer:
    return cast(AppContainer, request.app.state.container)


ContainerDep = Annotated[AppContainer, Depends(get_container)]
