# -*- coding: utf-8 -*-
"""Create a contact sheet and machine-readable QA summary for visual inspection."""

from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent


def main() -> None:
    images = [
        path
        for path in sorted(HERE.glob("Fig_S*.png"))
        if path.name != "Fig_S7_SHAP_multivariable_spatial_gain.png"
    ]
    thumbs = []
    rows = []
    for path in images:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            gray = np.asarray(rgb.convert("L").resize((256, 256)), dtype=float)
            rows.append({"file": path.name, "width": rgb.width, "height": rgb.height, "pixel_std": float(gray.std()), "near_white_fraction": float(np.mean(gray > 250))})
            copy = rgb.copy()
            copy.thumbnail((760, 500))
            canvas = Image.new("RGB", (780, 550), "white")
            canvas.paste(copy, ((780 - copy.width) // 2, 36 + (500 - copy.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), path.stem, fill="#222222")
            thumbs.append(canvas)
    cols = 2
    nrows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 780, nrows * 550), "white")
    for i, image in enumerate(thumbs):
        sheet.paste(image, ((i % cols) * 780, (i // cols) * 550))
    sheet.save(HERE / "QA_contact_sheet.png", dpi=(180, 180))
    (HERE / "Visual_QA_Summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    failed = [row for row in rows if row["width"] < 2500 or row["height"] < 1400 or row["pixel_std"] < 3]
    if failed:
        raise RuntimeError(f"Visual QA thresholds failed: {failed}")


if __name__ == "__main__":
    main()
