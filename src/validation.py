from pathlib import Path
from typing import Iterable, List, Optional

from .exceptions import ValidationError


def ensure_string(
    value,
    field_name: str,
    *,
    allow_empty: bool = False,
    max_length: Optional[int] = None,
) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string.")
    if not allow_empty and not value.strip():
        raise ValidationError(f"{field_name} must not be empty.")
    if max_length is not None and len(value) > max_length:
        raise ValidationError(f"{field_name} must be at most {max_length} characters.")
    return value


def ensure_api_path(value, field_name: str = "path", *, max_length: int = 512) -> str:
    path = ensure_string(value, field_name, max_length=max_length)
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValidationError(f"{field_name} must be a relative API path starting with '/'.")
    return path


def ensure_optional_string(value, field_name: str, *, max_length: Optional[int] = None) -> Optional[str]:
    if value is None:
        return None
    return ensure_string(value, field_name, max_length=max_length)


def ensure_optional_message_id(value, field_name: str, *, max_length: Optional[int] = None):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a non-empty string or positive integer.")
    if isinstance(value, int):
        if value <= 0:
            raise ValidationError(f"{field_name} must be a non-empty string or positive integer.")
        return value
    return ensure_string(value, field_name, max_length=max_length)


def ensure_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean.")
    return value


def ensure_positive_int(value, field_name: str, *, max_value: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValidationError(f"{field_name} must be a positive integer.")
    if max_value is not None and value > max_value:
        raise ValidationError(f"{field_name} must be at most {max_value}.")
    return value


def ensure_string_list(
    values,
    field_name: str,
    *,
    max_items: int = 100,
    max_item_length: int = 256,
) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValidationError(f"{field_name} must be a list of strings.")

    result = list(values)
    if len(result) > max_items:
        raise ValidationError(f"{field_name} must contain at most {max_items} items.")
    return [
        ensure_string(item, f"{field_name}[{index}]", max_length=max_item_length)
        for index, item in enumerate(result)
    ]


def ensure_existing_file(file_path: str) -> Path:
    path = Path(ensure_string(file_path, "file_path")).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationError(f"file_path does not exist: {path}") from exc

    if not resolved.is_file():
        raise ValidationError(f"file_path must point to a regular file: {resolved}")
    return resolved
