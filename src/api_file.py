import json
import base64
import mimetypes
import numbers
import time
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import FileInfo

from .http_client import DeepSeekHTTPClient
from .pow_solver import PoWSolver
from .exceptions import APIRequestError, FileProcessingTimeoutError, PoWSolverError, ValidationError
from .validation import ensure_bool, ensure_existing_file, ensure_positive_int, ensure_string


class FileAPI:
    UPLOAD_PATH = "/api/v0/file/upload_file"

    def __init__(self, http_client: DeepSeekHTTPClient, pow_solver: PoWSolver, max_upload_size_bytes: Optional[int] = None):
        self.http = http_client
        self.pow = pow_solver
        if max_upload_size_bytes is not None:
            ensure_positive_int(max_upload_size_bytes, "max_upload_size_bytes")
        self.max_upload_size_bytes = max_upload_size_bytes

    def fetch_files(self, file_ids: List[str]) -> dict:
        """
        Fetch status of uploaded files.

        Uses GET /api/v0/file/fetch_files?file_ids=id1,id2,...
        """
        if not file_ids:
            raise ValidationError("file_ids must contain at least one file id.")
        ids_str = ",".join(ensure_string(fid, "file_id", max_length=256) for fid in file_ids)
        return self.http.get("/api/v0/file/fetch_files", params={"file_ids": ids_str})

    def upload_file(self, file_path: str, wait_ready: bool = True, timeout: float = 30.0) -> dict:
        """
        Upload a file to DeepSeek.

        Requires PoW (Proof-of-Work), solved automatically.

        Args:
            file_path: Path to the file to upload.
            wait_ready: If True, poll until the file is processed and ready to use.
            timeout: Max seconds to wait for file processing (only if wait_ready=True).

        Returns:
            The API response containing the file info including ID and status.
            Use FileAPI.extract_file_info() to parse it.
        """
        wait_ready = ensure_bool(wait_ready, "wait_ready")
        if isinstance(timeout, bool) or not isinstance(timeout, numbers.Real) or timeout <= 0:
            raise ValidationError("timeout must be a positive number.")

        path = ensure_existing_file(file_path)

        file_size = path.stat().st_size
        if self.max_upload_size_bytes is not None and file_size > self.max_upload_size_bytes:
            raise ValidationError(
                f"file_path exceeds max_upload_size_bytes ({file_size} > {self.max_upload_size_bytes})."
            )

        # Solve PoW for the upload endpoint
        target_path = self.UPLOAD_PATH
        challenge_resp = self.http.post(
            "/api/v0/chat/create_pow_challenge",
            json_data={"target_path": target_path},
        )
        challenge = challenge_resp.get("data", {}).get("biz_data", {}).get("challenge")
        if not isinstance(challenge, dict):
            raise APIRequestError("Malformed PoW challenge response for file upload.")

        algorithm = challenge.get("algorithm", "DeepSeekHashV1")
        if algorithm != "DeepSeekHashV1":
            raise PoWSolverError(f"Unsupported PoW algorithm: {algorithm}")
        required_fields = ("challenge", "salt", "difficulty", "expire_at", "signature")
        missing = [field for field in required_fields if field not in challenge]
        if missing:
            raise APIRequestError(f"Malformed PoW challenge for file upload: missing {', '.join(missing)}.")

        nonce = self.pow.solve(
            challenge["challenge"], challenge["salt"], challenge["expire_at"], challenge["difficulty"]
        )
        solved = {
            "algorithm": algorithm,
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": nonce,
            "signature": challenge["signature"],
            "target_path": target_path,
        }
        pow_header = base64.b64encode(json.dumps(solved, separators=(",", ":")).encode()).decode("ascii")

        filename = path.name
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        try:
            with path.open("rb") as f:
                files = {"file": (filename, f, mime_type)}
                result = self.http.post(
                    self.UPLOAD_PATH,
                    files=files,
                    headers={
                        "x-ds-pow-response": pow_header,
                        "x-file-size": str(file_size),
                    },
                )
        except OSError as exc:
            raise ValidationError(f"Unable to read file_path: {path}") from exc

        if not wait_ready:
            return result

        # Poll until file is processed
        file_id = result.get("data", {}).get("biz_data", {}).get("id")
        if not file_id:
            raise APIRequestError("Malformed upload response: missing data.biz_data.id.")

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            try:
                status_resp = self.fetch_files([file_id])
                files_list = status_resp.get("data", {}).get("biz_data", {}).get("files", [])
                for info in files_list:
                    if info.get("id") == file_id:
                        status = info.get("status", "")
                        if status != "PENDING":
                            # Update result with latest file info
                            result["data"]["biz_data"] = info
                            return result
            except Exception as exc:
                raise APIRequestError(
                    f"Failed to poll uploaded file status for {file_id}: {exc.__class__.__name__}"
                ) from exc
            time.sleep(1)

        status = result.get("data", {}).get("biz_data", {}).get("status", "UNKNOWN")
        raise FileProcessingTimeoutError(
            f"Uploaded file {file_id} did not become ready within {float(timeout):.1f}s. Last status: {status}."
        )

    @staticmethod
    def extract_file_info(api_response: dict) -> "FileInfo":
        """Extract a FileInfo object from an upload_file or fetch_files response."""
        from .models import FileInfo
        
        biz_data = api_response.get("data", {}).get("biz_data", {})
        # Depending on the endpoint, the file might be directly in biz_data,
        # or inside a 'files' list.
        file_obj = biz_data
        if "files" in biz_data and isinstance(biz_data["files"], list) and len(biz_data["files"]) > 0:
            file_obj = biz_data["files"][0]
            
        return FileInfo(
            id=file_obj.get("id", ""),
            status=file_obj.get("status", "UNKNOWN"),
            file_name=file_obj.get("file_name", ""),
            file_size=file_obj.get("file_size", 0),
            preview_url=file_obj.get("signed_preview_url"),
            token_usage=file_obj.get("token_usage")
        )
