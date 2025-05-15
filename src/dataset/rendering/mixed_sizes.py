import itertools
import math
from typing import List, Tuple

import PIL.Image
import PIL.ImageDraw2
from PIL.ImageDraw2 import Pen


def resize(image: PIL.Image, max_size: int) -> PIL.Image:
    width, height = image.size[0], image.size[1]
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_width = int((width / height) * max_size)
            new_height = max_size
        image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
    return image


SIX_ELEMENT_PERMUTATIONS = list(itertools.permutations(range(6), 6))


def get_size(
    sizes: List[Tuple[int, int]], permutation: Tuple[int, ...]
) -> Tuple[int, int]:
    p = permutation
    width = max(
        [
            sizes[p[0]][0] + sizes[p[1]][0],
            sizes[p[2]][0] + sizes[p[3]][0],
            sizes[p[4]][0] + sizes[p[5]][0],
        ]
    )
    height = max(
        [
            sizes[p[0]][1] + sizes[p[2]][1] + sizes[p[4]][1],
            sizes[p[1]][1] + sizes[p[3]][1] + sizes[p[5]][1],
        ]
    )
    y1, y2 = 0, 0
    for i, p in enumerate(permutation):
        row = i // 2
        image_width, image_height = sizes[p]
        if i % 2 == 0:
            y = y1
            y1 += image_height
            # Check if top-right corner intersects with bottom-left corner from the right image in the previous row
            (image_width_right_prev_row, _) = sizes[permutation[i - 1]]
            if row > 0 and y < y2 and image_width > width - image_width_right_prev_row:
                additional_height = y2 - y
                height += additional_height
                y1 += additional_height
        else:
            y = y2
            y2 += image_height
            # Check if top-left corner intersects with bottom-right corner from the left image in the previous row
            (image_width_left_prev_row, _) = sizes[permutation[i - 3]]
            if row > 0 and y < y1 and width - image_width < image_width_left_prev_row:
                additional_height = y1 - y
                height += additional_height
                y2 += additional_height
    return width, height


def get_permutations(
    left_sizes: List[Tuple[int, int]], right_sizes: List[Tuple[int, int]]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    min_area = math.inf
    best_permutations = None
    for left_permutation in SIX_ELEMENT_PERMUTATIONS:
        for right_permutation in SIX_ELEMENT_PERMUTATIONS:
            left_width, left_height = get_size(left_sizes, left_permutation)
            right_width, right_height = get_size(right_sizes, right_permutation)
            width = left_width + right_width
            height = max(left_height, right_height)
            area = width * height
            if area < min_area:
                min_area = area
                best_permutations = (left_permutation, right_permutation)
    return best_permutations


def draw_side(
    margin: int,
    permutation: Tuple[int, ...],
    sizes: List[Tuple[int, int]],
    images: Tuple[PIL.Image.Image],
    background_color: str = "black",
) -> PIL.Image.Image:
    width, height = get_size(sizes, permutation)
    canvas = PIL.Image.new(
        "RGB", (margin + width, 2 * margin + height), color=background_color
    )
    y1, y2 = 0, 0
    for i, p in enumerate(permutation):
        row = i // 2
        image_width, image_height = sizes[p]
        image = images[p]
        x = 0 if i % 2 == 0 else width + margin - image_width
        if i % 2 == 0:
            y = y1
            y1 += image_height + margin
            # Check if top-right corner intersects with bottom-left corner from the right image in the previous row
            (image_width_right_prev_row, _) = sizes[permutation[i - 1]]
            if row > 0 and y < y2 and image_width > width - image_width_right_prev_row:
                additional_height = y2 - y
                height += additional_height
                y1 += additional_height
        else:
            y = y2
            y2 += image_height + margin
            # Check if top-left corner intersects with bottom-right corner from the left image in the previous row
            (image_width_left_prev_row, _) = sizes[permutation[i - 3]]
            if row > 0 and y < y1 and width - image_width < image_width_left_prev_row:
                additional_height = y1 - y
                height += additional_height
                y2 += additional_height
        canvas.paste(image, (x, y))
    return canvas


def draw_compact_bongard_problem(
    left_images: Tuple[PIL.Image.Image],
    right_images: Tuple[PIL.Image.Image],
    margin: int = 10,
    side_max_size: int = 512,
    max_size: int = 1024,
    background_color: str = "black",
    separator_color: str = "white",
) -> Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    left_sizes = [(image.size[0], image.size[1]) for image in left_images]
    right_sizes = [(image.size[0], image.size[1]) for image in right_images]

    left_permutation, right_permutation = get_permutations(left_sizes, right_sizes)

    left_canvas = draw_side(
        margin, left_permutation, left_sizes, left_images, background_color
    )
    right_canvas = draw_side(
        margin, right_permutation, right_sizes, right_images, background_color
    )

    left_width, left_height = (left_canvas.size[0], left_canvas.size[1])
    right_width, right_height = (right_canvas.size[0], right_canvas.size[1])

    total_width = left_width + 2 * margin + right_width
    total_height = max(left_height, right_height)
    canvas = PIL.Image.new("RGB", (total_width, total_height), color=background_color)
    canvas.paste(left_canvas, (0, 0))
    canvas.paste(right_canvas, (left_width + 2 * margin, 0))
    draw = PIL.ImageDraw2.Draw(canvas)
    draw.line(
        (left_width + margin, 0, left_width + margin, total_height),
        Pen(color=separator_color, width=margin // 2),
    )

    left_canvas = resize(left_canvas, side_max_size)
    right_canvas = resize(right_canvas, side_max_size)
    canvas = resize(canvas, max_size)

    return left_canvas, right_canvas, canvas
