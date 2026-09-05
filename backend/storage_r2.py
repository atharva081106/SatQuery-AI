import os
import uuid
import mimetypes
from typing import Optional

class R2StorageManager:
    """
    Zero-Egress Object Storage Client for Cloudflare R2 (S3-Compatible, 10GB free forever).
    Offloads raster masks and heatmaps so JSON responses stay small and fast.
    """
    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "satquery-artifacts")
        self.public_url = os.getenv("R2_PUBLIC_URL", "").rstrip("/")
        self._s3_client = None

    def is_configured(self) -> bool:
        return bool(self.account_id and self.access_key and self.secret_key)

    def _get_client(self):
        if not self._s3_client and self.is_configured():
            try:
                import boto3
                from botocore.config import Config
                endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
                self._s3_client = boto3.client(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    config=Config(signature_version="s3v4")
                )
            except Exception as e:
                print(f"[R2 Warning] boto3 initialization error: {e}")
                self._s3_client = None
        return self._s3_client

    def upload_bytes(self, data: bytes, filename: Optional[str] = None, content_type: str = "image/png") -> Optional[str]:
        """
        Uploads data bytes to R2 bucket.
        Returns public CDN URL if configured, or None to trigger base64 fallback.
        """
        if not self.is_configured():
            return None

        client = self._get_client()
        if not client:
            return None

        if not filename:
            ext = mimetypes.guess_extension(content_type) or ".png"
            filename = f"artifacts/{uuid.uuid4()}{ext}"

        try:
            client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=data,
                ContentType=content_type
            )
            if self.public_url:
                return f"{self.public_url}/{filename}"
            return f"https://{self.bucket_name}.r2.cloudflarestorage.com/{filename}"
        except Exception as err:
            print(f"[R2 Upload Error] {err}")
            return None

r2_storage = R2StorageManager()
