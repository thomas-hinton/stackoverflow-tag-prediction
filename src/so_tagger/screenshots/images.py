"""PNG files written by `docs-screenshots`: optimisation and orphan detection."""

from collections.abc import Iterable
from pathlib import Path

from so_tagger.screenshots.captures import Capture


def optimise_png(path: Path) -> int:
    """Quantise the PNG at ``path`` to a 256-colour palette in place; return its new size.

    Screenshots of documentation pages hold few colours, so a palette without dithering keeps
    text crisp and divides the size several times over. Needs Pillow (`docs` group).
    """
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
    palette = rgb.quantize(colors=256, dither=Image.Dither.NONE)
    palette.save(path, format="PNG", optimize=True)
    return path.stat().st_size


def orphan_images(images_dir: Path, captures: Iterable[Capture]) -> list[Path]:
    """PNG files in ``images_dir`` that no capture declares, sorted by name."""
    if not images_dir.is_dir():
        return []
    declared = {capture.file_name for capture in captures}
    return sorted(path for path in images_dir.glob("*.png") if path.name not in declared)
