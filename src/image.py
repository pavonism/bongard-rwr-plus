import os
from typing import Set, Optional

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png"}


def is_image_supported(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    ext = ext[1:]  # Remove leading dot: .jpg -> jpg
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def get_image_filename_with_extension(filename_without_extension: str) -> Optional[str]:
    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = f"{filename_without_extension}.{extension}"
        if os.path.exists(candidate):
            return candidate
    return None
