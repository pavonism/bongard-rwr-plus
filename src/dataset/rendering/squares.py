from typing import List, Tuple
from PIL import Image, ImageDraw2


def draw_side(
    margin: int,
    imgs_size: Tuple[int, int],
    images: Tuple[Image.Image],
    grid: Tuple[int, int],
    background_color: str = "black",
) -> Image.Image:
    canvas = Image.new(
        "RGB",
        (
            margin * (grid[0] - 1) + imgs_size[0] * grid[0],
            margin * (grid[1] - 1) + imgs_size[1] * grid[1],
        ),
        color=background_color,
    )

    cols = grid[0]

    for i, img in enumerate(images):
        img = img.resize(imgs_size, Image.Resampling.LANCZOS)
        col = i % cols
        row = i // cols

        x = col * (imgs_size[0] + margin)
        y = row * (imgs_size[1] + margin)

        canvas.paste(img, (x, y))

    return canvas


def draw_square_bongard_problem(
    left_images: List[Image.Image],
    right_images: List[Image.Image],
    margin: int = 10,
    side_size: Tuple[int, int] = (512, 1024),
    background_color: str = "black",
    separator_color: str = "white",
    grid: Tuple[int, int] = (2, 3),
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    img_width = (side_size[0] - grid[0] * margin) // grid[0]
    img_height = (side_size[1] - (grid[1] - 1) * margin) // grid[1]

    imgs_size = (min(img_width, img_height), min(img_width, img_height))

    left_canvas = draw_side(
        margin,
        imgs_size,
        left_images,
        grid,
        background_color,
    )

    right_canvas = draw_side(
        margin,
        imgs_size,
        right_images,
        grid,
        background_color,
    )

    left_width, left_height = (left_canvas.size[0], left_canvas.size[1])
    right_width, right_height = (right_canvas.size[0], right_canvas.size[1])

    total_width = left_width + 2 * margin + right_width
    total_height = max(left_height, right_height)
    canvas = Image.new("RGB", (total_width, total_height), color=background_color)
    canvas.paste(left_canvas, (0, 0))
    canvas.paste(right_canvas, (left_width + 2 * margin, 0))
    draw = ImageDraw2.Draw(canvas)
    separator_width = margin // 2
    separator_x = left_width + (2 * margin - separator_width) / 2
    draw.line(
        (separator_x, 0, separator_x, total_height),
        ImageDraw2.Pen(color=separator_color, width=margin // 2),
    )

    return left_canvas, right_canvas, canvas
