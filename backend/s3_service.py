import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Environment configuration
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "chest-xray-images")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# FLOCI_ENDPOINT_URL or AWS_ENDPOINT_URL (e.g. http://localhost:4566 for Floci emulator)
S3_ENDPOINT_URL = os.getenv("FLOCI_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "test")
ENABLE_S3 = os.getenv("ENABLE_S3_STORAGE", "true").lower() in ("true", "1", "yes")

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    if not ENABLE_S3:
        return None

    try:
        import boto3
        from botocore.config import Config

        config = Config(
            region_name=AWS_REGION,
            signature_version="s3v4",
            connect_timeout=2.0,
            read_timeout=3.0,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},  # Path style is best for Floci emulator
        )

        kwargs = {
            "service_name": "s3",
            "region_name": AWS_REGION,
            "aws_access_key_id": AWS_ACCESS_KEY,
            "aws_secret_access_key": AWS_SECRET_KEY,
            "config": config,
        }

        if S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = S3_ENDPOINT_URL
            logger.info("Connecting to S3 emulator (Floci/LocalStack) at %s", S3_ENDPOINT_URL)
        else:
            logger.info("Connecting to AWS Cloud S3 in region %s", AWS_REGION)

        _s3_client = boto3.client(**kwargs)
        return _s3_client
    except Exception as exc:
        logger.warning("Could not initialize S3 client: %s", exc)
        return None


def ensure_bucket_exists(bucket_name: str = S3_BUCKET_NAME) -> bool:
    client = get_s3_client()
    if not client:
        return False

    try:
        client.head_bucket(Bucket=bucket_name)
        return True
    except Exception:
        try:
            if AWS_REGION == "us-east-1":
                client.create_bucket(Bucket=bucket_name)
            else:
                client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
                )
            logger.info("Successfully created S3 bucket: %s", bucket_name)
            return True
        except Exception as create_exc:
            logger.warning("Could not ensure/create S3 bucket %s: %s", bucket_name, create_exc)
            return False


def upload_xray_to_s3(
    file_bytes: bytes,
    filename: str,
    content_type: str = "image/png",
    bucket_name: str = S3_BUCKET_NAME,
) -> Dict[str, Any]:
    """
    Uploads an uploaded chest X-ray image to S3 (either Floci emulator or AWS Cloud).
    Returns metadata including bucket, key, and accessible/presigned URL.
    """
    if not ENABLE_S3:
        return {
            "stored": False,
            "provider": "S3 storage disabled via config",
            "bucket": bucket_name,
        }

    client = get_s3_client()
    if not client:
        return {
            "stored": False,
            "provider": "Boto3 client unavailable",
            "bucket": bucket_name,
        }

    try:
        ensure_bucket_exists(bucket_name)

        clean_filename = os.path.basename(filename).replace(" ", "_")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        key = f"xrays/{datetime.utcnow().strftime('%Y/%m')}/{timestamp}_{unique_id}_{clean_filename}"

        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=file_bytes,
            ContentType=content_type,
            Metadata={
                "original_filename": clean_filename,
                "upload_time": timestamp,
                "source": "chest-xray-ai-app",
            },
        )

        is_floci = bool(S3_ENDPOINT_URL and ("4566" in S3_ENDPOINT_URL or "floci" in S3_ENDPOINT_URL or "localhost" in S3_ENDPOINT_URL))
        provider_name = "Floci (Local AWS Emulator)" if is_floci else "AWS S3 Cloud"

        # Generate presigned URL for secure access
        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": key},
                ExpiresIn=3600,
            )
        except Exception:
            if S3_ENDPOINT_URL:
                url = f"{S3_ENDPOINT_URL}/{bucket_name}/{key}"
            else:
                url = f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{key}"

        return {
            "stored": True,
            "provider": provider_name,
            "bucket": bucket_name,
            "key": key,
            "url": url,
            "region": AWS_REGION,
            "endpoint": S3_ENDPOINT_URL or f"https://s3.{AWS_REGION}.amazonaws.com",
            "sizeBytes": len(file_bytes),
            "uploadedAt": timestamp,
        }
    except Exception as exc:
        logger.warning("Failed to upload image to S3: %s", exc)
        return {
            "stored": False,
            "provider": "AWS S3 / Floci upload error",
            "error": str(exc),
            "bucket": bucket_name,
        }


def get_s3_status() -> Dict[str, Any]:
    """
    Returns current health and configuration status for S3 / Floci emulator.
    """
    client = get_s3_client()
    is_floci = bool(S3_ENDPOINT_URL and ("4566" in S3_ENDPOINT_URL or "floci" in S3_ENDPOINT_URL or "localhost" in S3_ENDPOINT_URL))
    status = {
        "enabled": ENABLE_S3,
        "provider": "Floci (Local AWS Emulator)" if is_floci else ("AWS S3 Cloud" if ENABLE_S3 else "Disabled"),
        "endpoint": S3_ENDPOINT_URL or "AWS Cloud Native",
        "bucket": S3_BUCKET_NAME,
        "region": AWS_REGION,
        "connected": False,
    }

    if client:
        try:
            client.head_bucket(Bucket=S3_BUCKET_NAME)
            status["connected"] = True
            status["bucketExists"] = True
        except Exception as e:
            status["connected"] = False
            status["error"] = str(e)

    return status
