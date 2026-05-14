import logging
import numbers
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    APIRequestError,
    AuthenticationError,
    JSONParseError,
    ValidationError,
    redact_sensitive_text,
)
from .validation import ensure_api_path, ensure_string


logger = logging.getLogger(__name__)
Timeout = Union[float, Tuple[float, float]]


DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 60.0)
DEFAULT_STREAM_TIMEOUT: Tuple[float, float] = (10.0, 300.0)
DEFAULT_ACCEPT_LANGUAGE = "en-US,en;q=0.9,vi;q=0.8"
DEFAULT_APP_VERSION = "2.0.0"
DEFAULT_CLIENT_LOCALE = "en_US"
DEFAULT_CLIENT_PLATFORM = "web"
DEFAULT_CLIENT_TIMEZONE_OFFSET = "25200"
DEFAULT_CLIENT_VERSION = "2.0.0"
DEFAULT_RETRY_METHODS = frozenset({"HEAD", "GET", "OPTIONS"})
DEFAULT_RETRY_STATUSES = (429, 500, 502, 503, 504)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _request_id(resp: requests.Response) -> Optional[str]:
    for header in ("x-request-id", "x-ds-request-id", "cf-ray"):
        value = resp.headers.get(header)
        if value:
            return value
    return None


def _response_text(resp: requests.Response) -> str:
    try:
        return resp.text
    except Exception as exc:
        return f"<failed to read response body: {exc.__class__.__name__}>"


def _validate_timeout(value: Timeout, field_name: str) -> Timeout:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValidationError(f"{field_name} must be a float or a (connect, read) tuple.")
        connect, read = value
        if isinstance(connect, bool) or isinstance(read, bool):
            raise ValidationError(f"{field_name} values must be positive numbers.")
        if not isinstance(connect, numbers.Real) or not isinstance(read, numbers.Real):
            raise ValidationError(f"{field_name} values must be positive numbers.")
        if connect <= 0 or read <= 0:
            raise ValidationError(f"{field_name} values must be positive.")
        return float(connect), float(read)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{field_name} must be a positive number or a (connect, read) tuple.")
    if value <= 0:
        raise ValidationError(f"{field_name} must be positive.")
    return float(value)


def _validate_headers(headers: Optional[Mapping[str, str]], field_name: str) -> Dict[str, str]:
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise ValidationError(f"{field_name} must be a mapping.")

    validated: Dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not key.strip():
            raise ValidationError(f"{field_name} header names must be non-empty strings.")
        if "\r" in key or "\n" in key:
            raise ValidationError(f"{field_name} header names must not contain newlines.")
        if value is None:
            continue
        if not isinstance(value, str):
            value = str(value)
        if "\r" in value or "\n" in value:
            raise ValidationError(f"{field_name} header {key!r} must not contain newlines.")
        validated[key] = value
    return validated


def _validate_retry_values(values: Iterable[int], field_name: str) -> Tuple[int, ...]:
    result = tuple(values)
    for value in result:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValidationError(f"{field_name} must contain positive integers.")
    return result


class DeepSeekHTTPClient:
    BASE_URL = "https://chat.deepseek.com"

    def __init__(
        self,
        token: Optional[str] = None,
        hif_dliq: Optional[str] = None,
        hif_leim: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        base_url: str = BASE_URL,
        timeout: Timeout = DEFAULT_TIMEOUT,
        stream_timeout: Timeout = DEFAULT_STREAM_TIMEOUT,
        max_retries: int = 2,
        session: Optional[requests.Session] = None,
        authorization: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        user_agent: Optional[str] = DEFAULT_USER_AGENT,
        accept_language: Optional[str] = DEFAULT_ACCEPT_LANGUAGE,
        app_version: Optional[str] = DEFAULT_APP_VERSION,
        client_locale: Optional[str] = DEFAULT_CLIENT_LOCALE,
        client_platform: Optional[str] = DEFAULT_CLIENT_PLATFORM,
        client_timezone_offset: Optional[Union[str, int]] = DEFAULT_CLIENT_TIMEZONE_OFFSET,
        client_version: Optional[str] = DEFAULT_CLIENT_VERSION,
        proxies: Optional[Mapping[str, str]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        trust_env: Optional[bool] = None,
        retry_methods: Iterable[str] = DEFAULT_RETRY_METHODS,
        retry_statuses: Iterable[int] = DEFAULT_RETRY_STATUSES,
        retry_backoff_factor: float = 0.5,
        request_options: Optional[Mapping[str, object]] = None,
    ):
        if not isinstance(base_url, str) or not base_url.startswith(("https://", "http://")):
            raise ValidationError("base_url must be an absolute HTTP(S) URL.")

        self.authorization = None
        self.token = None
        if authorization is not None:
            if not isinstance(authorization, str) or not authorization.strip():
                raise AuthenticationError("authorization must be a non-empty string when provided.")
            self.authorization = authorization.strip()
        elif token is not None:
            if not isinstance(token, str) or not token.strip():
                raise AuthenticationError("A valid userToken must be provided.")
            self.token = token.strip()
            self.authorization = f"Bearer {self.token}"

        self.default_headers = _validate_headers(default_headers, "default_headers")
        has_default_authorization = any(key.lower() == "authorization" for key in self.default_headers)
        if not self.authorization and not has_default_authorization:
            raise AuthenticationError("A valid userToken or authorization header must be provided.")

        self.base_url = base_url.rstrip("/")
        self.hif_dliq = hif_dliq
        self.hif_leim = hif_leim
        self.timeout = _validate_timeout(timeout, "timeout")
        self.stream_timeout = _validate_timeout(stream_timeout, "stream_timeout")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValidationError("max_retries must be a non-negative integer.")

        self.accept_language = None if accept_language is None else str(accept_language)
        self.app_version = None if app_version is None else str(app_version)
        self.client_locale = None if client_locale is None else str(client_locale)
        self.client_platform = None if client_platform is None else str(client_platform)
        self.client_timezone_offset = None if client_timezone_offset is None else str(client_timezone_offset)
        self.client_version = None if client_version is None else str(client_version)
        self.user_agent = user_agent
        self.proxies = dict(proxies) if proxies else None
        self.verify = verify
        self.cert = cert
        self.request_options = dict(request_options) if request_options else {}

        if isinstance(retry_backoff_factor, bool) or not isinstance(retry_backoff_factor, numbers.Real):
            raise ValidationError("retry_backoff_factor must be a non-negative number.")
        if retry_backoff_factor < 0:
            raise ValidationError("retry_backoff_factor must be a non-negative number.")
        retry_statuses = _validate_retry_values(retry_statuses, "retry_statuses")
        retry_methods = frozenset(str(method).upper() for method in retry_methods)

        self.session = session or requests.Session()
        if trust_env is not None:
            if not isinstance(trust_env, bool):
                raise ValidationError("trust_env must be a boolean when provided.")
            self.session.trust_env = trust_env

        if max_retries:
            retry = Retry(
                total=max_retries,
                connect=max_retries,
                read=max_retries,
                status=max_retries,
                backoff_factor=float(retry_backoff_factor),
                status_forcelist=retry_statuses,
                allowed_methods=retry_methods,
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

        if cookies:
            self.session.cookies.update(cookies)

    def _build_url(self, path: str) -> str:
        return f"{self.base_url}{ensure_api_path(path)}"

    def _headers(self, extra_headers: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "origin": self.base_url,
            "referer": f"{self.base_url}/",
        }
        if self.accept_language:
            headers["accept-language"] = self.accept_language
        if self.app_version:
            headers["x-app-version"] = self.app_version
        if self.client_locale:
            headers["x-client-locale"] = self.client_locale
        if self.client_platform:
            headers["x-client-platform"] = self.client_platform
        if self.client_timezone_offset:
            headers["x-client-timezone-offset"] = self.client_timezone_offset
        if self.client_version:
            headers["x-client-version"] = self.client_version
        if self.authorization:
            headers["authorization"] = self.authorization
        if self.user_agent:
            headers["user-agent"] = self.user_agent
        if self.hif_dliq:
            headers["x-hif-dliq"] = self.hif_dliq
        if self.hif_leim:
            headers["x-hif-leim"] = self.hif_leim
        headers = _validate_headers(headers, "base_headers")
        headers.update(self.default_headers)
        headers.update(_validate_headers(extra_headers, "headers"))
        return headers

    def _handle_response(self, resp: requests.Response, context: str) -> dict:
        request_id = _request_id(resp)
        if not resp.ok:
            body = _response_text(resp)
            logger.warning(
                "DeepSeek HTTP request failed",
                extra={"context": context, "status_code": resp.status_code, "request_id": request_id},
            )
            raise APIRequestError(
                f"HTTP request failed for {context}",
                status_code=resp.status_code,
                response_body=body,
                request_id=request_id,
            )
        try:
            data = resp.json()
        except ValueError as e:
            preview = redact_sensitive_text(_response_text(resp), limit=200)
            raise JSONParseError(f"Failed to parse JSON response from {context}: {e}. Body preview: {preview}") from e

        if not isinstance(data, dict):
            raise JSONParseError(f"Expected JSON object response from {context}, got {type(data).__name__}.")

        # DeepSeek specific API error checking
        api_code = data.get("code")
        if api_code != 0:
            msg = redact_sensitive_text(str(data.get("msg") or data.get("message") or "unknown"), limit=200)
            raise APIRequestError(
                f"DeepSeek API returned error code {api_code}: {msg}",
                status_code=resp.status_code,
                response_body=_response_text(resp),
                request_id=request_id,
                api_code=api_code,
            )

        # Check business-level errors (code=0 but biz_code!=0, e.g. rate limit)
        biz_code = (data.get("data") or {}).get("biz_code")
        if biz_code is not None and biz_code != 0:
            biz_msg = redact_sensitive_text(str((data.get("data") or {}).get("biz_msg") or "unknown"), limit=200)
            raise APIRequestError(
                f"DeepSeek business error {biz_code}: {biz_msg}",
                status_code=resp.status_code,
                response_body=_response_text(resp),
                request_id=request_id,
                api_code=biz_code,
            )

        return data

    def _request(
        self,
        method: str,
        path: str,
        request_options: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> requests.Response:
        url = self._build_url(path)
        context = f"{method.upper()} {path}"
        options = dict(self.request_options)
        if self.proxies is not None:
            options.setdefault("proxies", self.proxies)
        options.setdefault("verify", self.verify)
        if self.cert is not None:
            options.setdefault("cert", self.cert)
        if request_options:
            options.update(dict(request_options))
        options.update(kwargs)
        try:
            logger.debug("DeepSeek request started", extra={"context": context})
            return self.session.request(method, url, **options)
        except requests.RequestException as exc:
            logger.warning(
                "DeepSeek network request failed",
                extra={"context": context, "error_type": exc.__class__.__name__},
            )
            raise APIRequestError(f"Network error during {context}: {exc.__class__.__name__}") from exc

    def get(
        self,
        path: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        request_options: Optional[Mapping[str, object]] = None,
    ) -> dict:
        resp = self._request(
            "GET",
            path,
            params=params,
            headers=self._headers(headers),
            timeout=self.timeout,
            request_options=request_options,
        )
        return self._handle_response(resp, f"GET {path}")

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        data=None,
        files=None,
        headers: Optional[dict] = None,
        stream: bool = False,
        timeout: Optional[Timeout] = None,
        request_options: Optional[Mapping[str, object]] = None,
        **kwargs,
    ) -> requests.Response:
        """Send a raw request and return the underlying requests.Response."""
        method = ensure_string(method, "method", max_length=16).upper()
        req_headers = self._headers(headers)
        if files:
            for key in list(req_headers):
                if key.lower() == "content-type":
                    del req_headers[key]
        effective_timeout = timeout if timeout is not None else (self.stream_timeout if stream else self.timeout)
        effective_timeout = _validate_timeout(effective_timeout, "timeout")
        return self._request(
            method,
            path,
            params=params,
            json=json_data,
            data=data,
            files=files,
            headers=req_headers,
            stream=stream,
            timeout=effective_timeout,
            request_options=request_options,
            **kwargs,
        )

    def post(
        self,
        path: str,
        json_data: Optional[dict] = None,
        files=None,
        headers: Optional[dict] = None,
        request_options: Optional[Mapping[str, object]] = None,
    ) -> dict:
        req_headers = self._headers(headers)
        if files:
            # requests injects the multipart boundary only when content-type is unset.
            for key in list(req_headers):
                if key.lower() == "content-type":
                    del req_headers[key]

        resp = self._request(
            "POST",
            path,
            json=json_data,
            files=files,
            headers=req_headers,
            timeout=self.timeout,
            request_options=request_options,
        )
        return self._handle_response(resp, f"POST {path}")

    def post_stream(
        self,
        path: str,
        json_data: dict,
        headers: Optional[dict] = None,
        request_options: Optional[Mapping[str, object]] = None,
    ) -> requests.Response:
        resp = self._request(
            "POST",
            path,
            json=json_data,
            headers=self._headers(headers),
            stream=True,
            timeout=self.stream_timeout,
            request_options=request_options,
        )
        if not resp.ok:
            body = _response_text(resp)
            request_id = _request_id(resp)
            resp.close()
            raise APIRequestError(
                f"Streaming request failed for POST {path}",
                status_code=resp.status_code,
                response_body=body,
                request_id=request_id,
            )
        return resp

    def set_default_header(self, name: str, value: str) -> None:
        self.default_headers.update(_validate_headers({name: value}, "default_headers"))

    def remove_default_header(self, name: str) -> None:
        for key in list(self.default_headers):
            if key.lower() == name.lower():
                del self.default_headers[key]

    def set_cookie(self, name: str, value: str) -> None:
        self.session.cookies.set(name, value)

    def clear_cookies(self) -> None:
        self.session.cookies.clear()

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
