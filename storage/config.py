"""
Object-store selection from environment (M-storage).

One env var flips the whole system between local filesystem and S3-compatible
cloud storage, without touching any calling code:

    KAFUL_OBJECT_STORE = "fs" (default) | "s3"

For s3 (works with AWS S3, Cloudflare R2, Backblaze B2, MinIO):
    KAFUL_S3_BUCKET        (required)
    KAFUL_S3_PREFIX        (default "kaful")
    KAFUL_S3_ENDPOINT_URL  (omit for native AWS S3; set for R2/B2/MinIO)
    KAFUL_S3_REGION        (optional)
Credentials use the standard AWS chain (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
"""

from __future__ import annotations

import os
from pathlib import Path

from storage.filesystem_object_store import FilesystemObjectStore


def object_store_from_env(store_dir: str | os.PathLike):
    """Build the configured ObjectStore. Defaults to local filesystem."""
    kind = os.environ.get("KAFUL_OBJECT_STORE", "fs").lower()
    if kind == "s3":
        from storage.s3_object_store import S3ObjectStore
        bucket = os.environ.get("KAFUL_S3_BUCKET")
        if not bucket:
            raise RuntimeError("KAFUL_OBJECT_STORE=s3 but KAFUL_S3_BUCKET is not set")
        return S3ObjectStore(
            bucket,
            prefix=os.environ.get("KAFUL_S3_PREFIX", "kaful"),
            endpoint_url=os.environ.get("KAFUL_S3_ENDPOINT_URL") or None,
            region=os.environ.get("KAFUL_S3_REGION") or None,
        )
    return FilesystemObjectStore(Path(store_dir) / "object_store")
