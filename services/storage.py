import uuid
from pathlib import Path

from config import settings


def get_public_url(storage_key: str) -> str:
    if storage_key.startswith("http://") or storage_key.startswith("https://"):
        return storage_key
    if settings.storage_backend == "local":
        return f"{settings.local_files_url}/{storage_key}"
    protocol = "https" if settings.minio_secure else "http"
    return f"{protocol}://{settings.minio_public_endpoint}/{settings.minio_bucket}/{storage_key}"


def _local_path(storage_key: str) -> Path:
    path = settings.local_storage_dir / storage_key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def save_local_file(filename: str, content: bytes, content_type: str) -> dict:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    storage_key = f"assets/{uuid.uuid4()}.{ext}"
    path = _local_path(storage_key)
    path.write_bytes(content)
    return {
        "storage_key": storage_key,
        "public_url": get_public_url(storage_key),
        "mime_type": content_type,
    }


def generate_presigned_upload(filename: str, content_type: str) -> dict:
    """Legacy presign endpoint — local mode uses direct /upload/file instead."""
    if settings.storage_backend == "local":
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        storage_key = f"assets/{uuid.uuid4()}.{ext}"
        return {
            "storage_key": storage_key,
            "upload_url": f"{settings.local_files_url.replace('/files', '')}/upload/file",
            "fields": {},
            "public_url": get_public_url(storage_key),
            "local_mode": True,
        }

    import boto3
    from botocore.client import Config

    protocol = "https" if settings.minio_secure else "http"
    client = boto3.client(
        "s3",
        endpoint_url=f"{protocol}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
    storage_key = f"assets/{uuid.uuid4()}.{ext}"
    presigned = client.generate_presigned_post(
        Bucket=settings.minio_bucket,
        Key=storage_key,
        Fields={"Content-Type": content_type},
        Conditions=[{"Content-Type": content_type}],
        ExpiresIn=3600,
    )
    return {
        "storage_key": storage_key,
        "upload_url": presigned["url"],
        "fields": presigned["fields"],
        "public_url": get_public_url(storage_key),
        "local_mode": False,
    }
