"""Ingestion routes — upload and process documents."""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import get_vector_store, rate_limiter
from src.api.schemas import IngestResponse
from src.core.config import settings
from src.core.logging import logger
from src.ingestion.loader import SUPPORTED_EXTENSIONS
from src.ingestion.pipeline import ingest_files

router = APIRouter(prefix="/api/v1", tags=["ingestion"])

_MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024
_COLLECTION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    collection: str = Form(default="default"),
    _api_key: str = Depends(rate_limiter),
):
    """Upload and ingest documents into the vector store."""
    # Validate collection name
    if not _COLLECTION_PATTERN.match(collection):
        raise HTTPException(status_code=400, detail="Invalid collection name")

    task_id = str(uuid.uuid4())
    logger.info("Ingestion task %s started with %d file(s)", task_id, len(files))

    temp_dir = Path(tempfile.mkdtemp(prefix="rag_ingest_"))
    saved_paths: list[Path] = []

    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file: %s", upload.filename)
                continue

            content = await upload.read()

            # Enforce file size limit
            if len(content) > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{upload.filename}' exceeds {settings.max_upload_size_mb}MB limit",
                )

            # Sanitize filename — strip path separators
            safe_name = Path(upload.filename or f"file_{uuid.uuid4()}{suffix}").name
            dest = temp_dir / safe_name
            dest.write_bytes(content)
            saved_paths.append(dest)

        if not saved_paths:
            return IngestResponse(
                task_id=task_id,
                status="failed",
                message="No supported files provided",
            )

        vector_store = get_vector_store()
        summary = await ingest_files(saved_paths, vector_store, collection=collection)

        return IngestResponse(
            task_id=task_id,
            status="completed",
            documents_loaded=summary["documents_loaded"],
            chunks_created=summary["chunks_created"],
            chunks_stored=summary["chunks_stored"],
            message=f"Successfully ingested {len(saved_paths)} file(s)",
        )

    except HTTPException:
        raise  # Let FastAPI handle 4xx errors directly
    except Exception as exc:
        logger.exception("Ingestion task %s failed", task_id)
        return IngestResponse(
            task_id=task_id,
            status="failed",
            message=str(exc),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
