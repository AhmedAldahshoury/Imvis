from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out = Path("sample_images")
    out.mkdir(parents=True, exist_ok=True)

    palette = [
        (220, 20, 60),
        (30, 144, 255),
        (46, 139, 87),
        (238, 130, 238),
        (255, 165, 0),
        (70, 70, 70),
    ]
    for idx, color in enumerate(palette, start=1):
        image = Image.new("RGB", (640, 420), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 30, 610, 390), outline=(255, 255, 255), width=6)
        draw.text((48, 48), f"Sample {idx}", fill=(255, 255, 255))
        image.save(out / f"sample_{idx}.jpg", quality=90)


if __name__ == "__main__":
    main()
