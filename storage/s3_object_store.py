"""
S3ObjectStore — an ObjectStore backed by any S3-compatible service.

Same put/get/exists-by-key contract as FilesystemObjectStore, so it drops in with
zero changes to the ingest handler, twin, or API — the payoff of the ObjectStore
abstraction (domain/stores.py). One implementation serves all S3-compatible
backends; only endpoint_url + credentials differ:

  * AWS S3        — endpoint_url=None (native), region set, standard AWS creds.
  * Cloudflare R2 — endpoint_url="https://<acct>.r2.cloudflarestorage.com".
  * Backblaze B2  — endpoint_url="https://s3.<region>.backblazeb2.com".
  * MinIO         — endpoint_url="http://localhost:9000".

Credentials come from the standard boto3 chain (env vars AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, or an instance role) — never hardcoded.
"""

from __future__ import annotations

from domain.stores import ObjectStore


class S3ObjectStore(ObjectStore):
    def __init__(self, bucket: str, *, prefix: str = "", endpoint_url: str | None = None,
                 region: str | None = None, client=None) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is not None:
            self._s3 = client
        else:
            import boto3  # imported lazily so the fs-only install needs no boto3
            self._s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def _key(self, key: str) -> str:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(f"unsafe object key: {key!r}")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> None:
        full = self._key(key)   # validate before any client I/O
        self._s3.put_object(Bucket=self.bucket, Key=full, Body=data)

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError
        full = self._key(key)   # validate before any client I/O
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=full)
            return resp["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchBucket"):
                raise KeyError(key) from None
            raise

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        full = self._key(key)   # validate before any client I/O
        try:
            self._s3.head_object(Bucket=self.bucket, Key=full)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
