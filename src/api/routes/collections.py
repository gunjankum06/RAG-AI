"""Collection management routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_vector_store, rate_limiter
from src.api.schemas import CollectionSchema

router = APIRouter(prefix="/api/v1", tags=["collections"])


@router.get("/collections", response_model=list[CollectionSchema])
async def list_collections(_api_key: str = Depends(rate_limiter)):
    """List all collections with their document counts."""
    store = get_vector_store()
    collections = await store.list_collections()
    return [CollectionSchema(**c) for c in collections]


@router.delete("/collections/{name}")
async def delete_collection(name: str, _api_key: str = Depends(rate_limiter)):
    """Delete a collection and all its vectors."""
    store = get_vector_store()
    await store.delete_collection(name)
    return {"status": "deleted", "collection": name}
