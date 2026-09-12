#!/usr/bin/env python3
"""
trim_gifs.py
------------
Reduces file size of pick-place rollout GIFs by:
  1. Downscaling to 50% of original dimensions
  2. Keeping every other frame (halves frame count)
  3. Re-quantising to 64 colours (GIF palette)

Run once before pushing if GIFs are >3 MB each:
    python trim_gifs.py
"""

from pathlib import Path
from PIL import Image, ImageSequence

GIF_DIR = Path("models")
TARGET_GIFS = [
    "pick_place_rollout_1.gif",
    "pick_place_rollout_2.gif",
    "pick_place_rollout_3.gif",
]
SCALE = 0.5          # linear dimension scale (0.5 = quarter the pixels)
KEEP_EVERY_N = 2     # keep 1 frame out of every N
PALETTE_COLOURS = 64 # max GIF palette size (256 max)


def trim_gif(src: Path, dst: Path):
    img = Image.open(src)
    frames = []
    durations = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i % KEEP_EVERY_N != 0:
            continue
        w, h = frame.size
        new_w, new_h = int(w * SCALE), int(h * SCALE)
        resized = frame.convert("RGBA").resize((new_w, new_h), Image.LANCZOS)
        # GIF palette quantisation requires an RGB (not RGBA) input for MEDIANCUT
        background = Image.new("RGB", resized.size, (255, 255, 255))
        background.paste(resized, mask=resized.split()[3])  # apply alpha
        quantised = background.quantize(colors=PALETTE_COLOURS)
        frames.append(quantised)
        d = frame.info.get("duration", 50)
        durations.append(d * KEEP_EVERY_N)  # preserve apparent speed

    if not frames:
        print(f"  SKIP {src.name} — no frames")
        return

    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
        optimize=True,
    )
    before_kb = src.stat().st_size / 1024
    after_kb  = dst.stat().st_size / 1024
    print(f"  {src.name}: {before_kb:.0f} KB -> {after_kb:.0f} KB ({after_kb/before_kb:.0%})")


if __name__ == "__main__":
    for name in TARGET_GIFS:
        src = GIF_DIR / name
        if not src.exists():
            print(f"  SKIP {name} — not found")
            continue
        trim_gif(src, src)  # overwrite in-place
    print("Done.")
