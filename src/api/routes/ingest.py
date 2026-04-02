"""Ingestion routes — upload and process documents."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.api.dependencies import get_vector_store, rate_limiter
from src.api.schemas import IngestResponse
from src.core.logging import logger
from src.ingestion.loader import SUPPORTED_EXTENSIONS
from src.ingestion.pipeline import ingest_files

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    collection: str = Form(default="default"),
    _api_key: str = Depends(rate_limiter),
):
    """Upload and ingest documents into the vector store."""
    task_id = str(uuid.uuid4())
    logger.info("Ingestion task %s started with %d file(s)", task_id, len(files))

    # Save uploaded files to a temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_ingest_"))
    saved_paths: list[Path] = []

    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file: %s", upload.filename)
                continue

            dest = temp_dir / (upload.filename or f"file_{uuid.uuid4()}{suffix}")
            content = await upload.read()
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

    except Exception as exc:
        logger.exception("Ingestion task %s failed", task_id)
        return IngestResponse(
            task_id=task_id,
            status="failed",
            message=str(exc),
        )
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
