import os
from typing import Set, Optional

import PIL.Image

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png"}


class Content:
    def __init__(self):
        pass


class TextContent(Content):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __str__(self) -> str:
        return f"{self.__class__.__name__} - text: {self.text}"


class ImageContent(Content):
    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    @staticmethod
    def from_basename(basename: str) -> "ImageContent":
        filename = get_image_filename_with_extension(basename)
        return ImageContent(filename)

    def to_pil_image(self) -> PIL.Image:
        return PIL.Image.open(self.image_path)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} - image_path: {self.image_path}"


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
