"""Shared multipart file reading and size checks for upload routes."""

from fastapi import UploadFile

from app.api.deps import ContainerDep
from app.core.exceptions import InvalidInputError


async def read_upload(container: ContainerDep, upload: UploadFile) -> bytes:
    data = await upload.read()
    if not data:
        raise InvalidInputError("Uploaded file is empty.")
    if len(data) > container.settings.max_file_size_bytes:
        raise InvalidInputError(
            f"File exceeds the {container.settings.max_file_size_mb} MB size limit."
        )
    return data
