#!/usr/bin/env python3
"""Build exact-pixel BSDS500 qualitative tables for five selected pairs."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CLEAN = Path(os.environ.get("BSDS500_CLEAN_ROOT", ROOT / "data/BSDS500/images/test"))
GENDEG = ROOT / "results/bsds500_vanilla_gendeg_ordered_60ep_seed2026/generated/order2"
OURS = ROOT / "results/20261019_112259_KST_bsds500_gendeg_init_no_llr_512_epoch030_cfg1x1_7img/order2"
OURS_EPOCH4 = ROOT / "results/bsds500_epoch004_factor_tokens_physical_prior_selected5_seed2031/order2"
ONERESTORE = ROOT / "results/other_models/bsds7_cdd11_generation"
OUT = ROOT / "results/bsds500_selected_5factor_comparison"

MATPLOTLIB_FONTS = Path(
    os.environ.get("MATPLOTLIB_FONT_DIR", ROOT / "assets/fonts")
)
# STIXGeneral is the installed Times-compatible publication font.
FONT = MATPLOTLIB_FONTS / "STIXGeneral.ttf"
FONT_BOLD = MATPLOTLIB_FONTS / "STIXGeneralBol.ttf"
FONT_SERIF = FONT


@dataclass(frozen=True)
class Row:
    label: str
    image_id: str
    first: str
    second: str
    ours_category: str
    onerestore_category: str


ROWS = (
    Row("Haze + Snow", "41029", "haze", "snow", "haze_snow", "snow__to__haze"),
    Row("Low-light + Haze", "123057", "low-light", "haze", "low_haze", "low__to__haze"),
    Row("Low-light + Snow", "285022", "low-light", "snow", "low_snow", "low__to__snow"),
    Row("Low-light + Rain", "16068", "low-light", "rain", "low_rain", "low__to__rain"),
    Row("Haze + Rain", "290035", "haze", "rain", "haze_rain", "rain__to__haze"),
)

DISPLAY = {"low-light": "Low-light", "haze": "Haze", "rain": "Rain", "snow": "Snow"}
TILE = 360
LABEL_WIDTH = 220
CAPTION_HEIGHT = 50
ROW_HEIGHT = TILE + CAPTION_HEIGHT
TITLE_HEIGHT = 64
NOTE_HEIGHT = 44
HEADER_HEIGHT = 82
PANEL_LABEL_HEIGHT = 62


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)
    except OSError:
        return ImageFont.load_default()


def serif_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(FONT_SERIF), size)
    except OSError:
        return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
             *, size: int, bold: bool = False, fill: str = "#172433") -> None:
    x0, y0, x1, y1 = box
    fnt = font(size, bold)
    lines = text.split("\n")
    bounds = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    line_heights = [bound[3] - bound[1] for bound in bounds]
    total_height = sum(line_heights) + 5 * max(0, len(lines) - 1)
    y = y0 + (y1 - y0 - total_height) // 2
    for line, bound, line_height in zip(lines, bounds, line_heights):
        line_width = bound[2] - bound[0]
        draw.text((x0 + (x1 - x0 - line_width) // 2, y), line, font=fnt, fill=fill)
        y += line_height + 5


def exact_square(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.fit(
        image, (TILE, TILE), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )


def exact_fit(path: Path, size: tuple[int, int]) -> Image.Image:
    """Crop once from the source and render directly at the requested cell size."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.fit(
        image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )


def exact_square_at(path: Path, side: int) -> Image.Image:
    """Use the same centered square crop as the original portrait table."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.fit(
        image, (side, side), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )


def row_sources(row: Row) -> dict[str, tuple[Path, str]]:
    first_to_second = f"{row.first}__to__{row.second}"
    second_to_first = f"{row.second}__to__{row.first}"
    return {
        "clean": (CLEAN / f"{row.image_id}.jpg", f"BSDS500: {row.image_id}"),
        "onerestore": (
            ONERESTORE / row.onerestore_category / f"{row.image_id}.png",
            f"CDD-11: {DISPLAY[row.first]} + {DISPLAY[row.second]}",
        ),
        "order1": (
            GENDEG / first_to_second / f"{row.image_id}.png",
            f"{DISPLAY[row.first]} → {DISPLAY[row.second]}",
        ),
        "order2": (
            GENDEG / second_to_first / f"{row.image_id}.png",
            f"{DISPLAY[row.second]} → {DISPLAY[row.first]}",
        ),
        "ours": (
            OURS / row.ours_category / f"{row.image_id}.png",
            f"Joint: {DISPLAY[row.first]} + {DISPLAY[row.second]}",
        ),
        "ours_epoch4": (
            OURS_EPOCH4 / row.ours_category / f"{row.image_id}.png",
            f"Joint: {DISPLAY[row.first]} + {DISPLAY[row.second]}",
        ),
    }


def build(include_onerestore: bool) -> Path:
    columns = ["clean"]
    if include_onerestore:
        columns.append("onerestore")
    columns.extend(("order1", "order2", "ours"))
    headers = {
        "clean": "Clean\n(BSDS500)",
        "onerestore": "OneRestore\n(CDD-11 synthesis)",
        "order1": "GenDeg\nOrder 1",
        "order2": "GenDeg\nOrder 2",
        "ours": "Ours\n(Joint 1-Pass)",
    }

    width = LABEL_WIDTH + TILE * len(columns)
    panel_label_height = 0 if include_onerestore else PANEL_LABEL_HEIGHT
    height = TITLE_HEIGHT + NOTE_HEIGHT + HEADER_HEIGHT + ROW_HEIGHT * len(ROWS) + panel_label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, width, TITLE_HEIGHT), fill="#172433")
    draw.text(
        (18, 15), "BSDS500 selected 2-factor degradation comparison",
        font=font(32, True), fill="white",
    )
    draw.rectangle((0, TITLE_HEIGHT, width, TITLE_HEIGHT + NOTE_HEIGHT), fill="#e9eef3")
    note = "Same source per row | GenDeg: sequential 2-pass | Ours: simultaneous joint condition"
    draw.text((15, TITLE_HEIGHT + 9), note, font=font(20), fill="#263746")

    header_y = TITLE_HEIGHT + NOTE_HEIGHT
    draw.rectangle(
        (0, header_y, LABEL_WIDTH, header_y + HEADER_HEIGHT),
        fill="#dbe7f2", outline="#aab7c2",
    )
    centered(draw, (0, header_y, LABEL_WIDTH, header_y + HEADER_HEIGHT),
             "Degradation" if not include_onerestore else "Degradation\n(Image ID)",
             size=24 if not include_onerestore else 21, bold=True)
    for index, column in enumerate(columns):
        x0 = LABEL_WIDTH + index * TILE
        draw.rectangle(
            (x0, header_y, x0 + TILE, header_y + HEADER_HEIGHT),
            fill="#dbe7f2", outline="#aab7c2",
        )
        centered(draw, (x0, header_y, x0 + TILE, header_y + HEADER_HEIGHT),
                 headers[column], size=22, bold=True)

    for row_index, row in enumerate(ROWS):
        y0 = header_y + HEADER_HEIGHT + row_index * ROW_HEIGHT
        background = "#ffffff" if row_index % 2 == 0 else "#edf3f8"
        draw.rectangle((0, y0, width, y0 + ROW_HEIGHT), fill=background, outline="#b8c1ca")
        centered(
            draw, (0, y0, LABEL_WIDTH, y0 + ROW_HEIGHT),
            (row.label.replace(" + ", " +\n") if not include_onerestore
             else f"{row.label}\n({row.image_id})"),
            size=30 if not include_onerestore else 23, bold=True,
        )
        sources = row_sources(row)
        for column_index, column in enumerate(columns):
            x0 = LABEL_WIDTH + column_index * TILE
            path, caption = sources[column]
            if not include_onerestore and column == "clean":
                caption = "Clean"
            canvas.paste(exact_square(path), (x0, y0))
            draw.rectangle(
                (x0, y0 + TILE, x0 + TILE, y0 + ROW_HEIGHT),
                fill="#eef3f7", outline="#b8c1ca",
            )
            centered(
                draw, (x0, y0 + TILE, x0 + TILE, y0 + ROW_HEIGHT),
                caption, size=23 if not include_onerestore else 18,
                bold=column in {"order1", "order2", "ours"},
                fill="#173f8f" if column in {"order1", "order2", "ours"} else "#394a5a",
            )

    if not include_onerestore:
        label_y = header_y + HEADER_HEIGHT + ROW_HEIGHT * len(ROWS)
        draw.rectangle((0, label_y, width, height), fill="white")
        for column_index, panel_label in enumerate(("(a)", "(b)", "(c)", "(d)")):
            x0 = LABEL_WIDTH + column_index * TILE
            panel_font = serif_font(36)
            bound = draw.textbbox((0, 0), panel_label, font=panel_font)
            text_width = bound[2] - bound[0]
            text_height = bound[3] - bound[1]
            draw.text(
                (x0 + (TILE - text_width) // 2,
                 label_y + (PANEL_LABEL_HEIGHT - text_height) // 2 - bound[1]),
                panel_label, font=panel_font, fill="#111111",
            )

    OUT.mkdir(parents=True, exist_ok=True)
    name = "selected_5pairs_with_onerestore_5col.png" if include_onerestore else "selected_5pairs_4col.png"
    destination = OUT / name
    canvas.save(destination, optimize=True)
    return destination


def build_landscape_4x3() -> Path:
    """Render the four-method poster panel natively at 3200x2400 (4:3)."""
    columns = ("clean", "order1", "order2", "ours")
    headers = {
        "clean": "Clean\n(BSDS500)",
        "order1": "GenDeg\nOrder 1",
        "order2": "GenDeg\nOrder 2",
        "ours": "Ours\n(Joint 1-Pass)",
    }
    canvas_width, canvas_height = 3200, 2400
    label_width = 360
    tile_width = (canvas_width - label_width) // len(columns)  # 710 px
    header_height = 120
    panel_height = 90
    row_height = (canvas_height - header_height - panel_height) // len(ROWS)  # 438 px
    caption_height = 62
    image_height = row_height - caption_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle(
        (0, 0, label_width, header_height), fill="#dbe7f2", outline="#aab7c2", width=2
    )
    centered(draw, (0, 0, label_width, header_height), "Degradation", size=42, bold=True)
    for column_index, column in enumerate(columns):
        x0 = label_width + column_index * tile_width
        draw.rectangle(
            (x0, 0, x0 + tile_width, header_height),
            fill="#dbe7f2", outline="#aab7c2", width=2,
        )
        centered(
            draw, (x0, 0, x0 + tile_width, header_height),
            headers[column], size=40, bold=True,
        )

    for row_index, row in enumerate(ROWS):
        y0 = header_height + row_index * row_height
        background = "#ffffff" if row_index % 2 == 0 else "#edf3f8"
        draw.rectangle(
            (0, y0, canvas_width, y0 + row_height),
            fill=background, outline="#b8c1ca", width=2,
        )
        centered(
            draw, (0, y0, label_width, y0 + row_height),
            row.label.replace(" + ", " +\n"), size=45, bold=True,
        )
        sources = row_sources(row)
        for column_index, column in enumerate(columns):
            x0 = label_width + column_index * tile_width
            path, caption = sources[column]
            if column == "clean":
                caption = "Clean"
            # Preserve the exact square field of view used in the original table;
            # only map that finished panel into the requested landscape cell.
            panel = exact_square_at(path, 720).resize(
                (tile_width, image_height), Image.Resampling.LANCZOS
            )
            canvas.paste(panel, (x0, y0))
            draw.rectangle(
                (x0, y0 + image_height, x0 + tile_width, y0 + row_height),
                fill="#eef3f7", outline="#b8c1ca", width=2,
            )
            centered(
                draw, (x0, y0 + image_height, x0 + tile_width, y0 + row_height),
                caption, size=36, bold=column in {"order1", "order2", "ours"},
                fill="#173f8f" if column in {"order1", "order2", "ours"} else "#394a5a",
            )

    label_y = header_height + row_height * len(ROWS)
    draw.rectangle((0, label_y, canvas_width, canvas_height), fill="white")
    for column_index, panel_label in enumerate(("(a)", "(b)", "(c)", "(d)")):
        x0 = label_width + column_index * tile_width
        panel_font = serif_font(52)
        bound = draw.textbbox((0, 0), panel_label, font=panel_font)
        text_width = bound[2] - bound[0]
        text_height = bound[3] - bound[1]
        draw.text(
            (x0 + (tile_width - text_width) // 2,
             label_y + (panel_height - text_height) // 2 - bound[1]),
            panel_label, font=panel_font, fill="#111111",
        )

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "selected_5pairs_4col_landscape_4x3.png"
    canvas.save(destination, optimize=True, dpi=(300, 300))
    return destination


def build_landscape_epoch4_5col() -> Path:
    """Add the epoch-4 factor-token/preservation-loss model to the paper table."""
    columns = ("clean", "order1", "order2", "ours", "ours_epoch4")
    headers = {
        "clean": "Clean\n(BSDS500)",
        "order1": "GenDeg\nOrder 1",
        "order2": "GenDeg\nOrder 2",
        "ours": "Ours\n(Joint 1-Pass)",
        "ours_epoch4": "Ours · Epoch 4\nFactor tokens +\npreservation loss",
    }
    canvas_width, canvas_height = 3910, 2400
    label_width, tile_width = 360, 710
    header_height, panel_height = 120, 90
    row_height = (canvas_height - header_height - panel_height) // len(ROWS)
    caption_height = 62
    image_height = row_height - caption_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (0, 0, label_width, header_height), fill="#dbe7f2", outline="#aab7c2", width=2
    )
    centered(draw, (0, 0, label_width, header_height), "Degradation", size=42, bold=True)
    for column_index, column in enumerate(columns):
        x0 = label_width + column_index * tile_width
        draw.rectangle(
            (x0, 0, x0 + tile_width, header_height),
            fill="#dbe7f2", outline="#aab7c2", width=2,
        )
        centered(
            draw, (x0, 0, x0 + tile_width, header_height), headers[column],
            size=34 if column == "ours_epoch4" else 40, bold=True,
        )

    for row_index, row in enumerate(ROWS):
        y0 = header_height + row_index * row_height
        background = "#ffffff" if row_index % 2 == 0 else "#edf3f8"
        draw.rectangle(
            (0, y0, canvas_width, y0 + row_height),
            fill=background, outline="#b8c1ca", width=2,
        )
        centered(
            draw, (0, y0, label_width, y0 + row_height),
            row.label.replace(" + ", " +\n"), size=45, bold=True,
        )
        sources = row_sources(row)
        for column_index, column in enumerate(columns):
            x0 = label_width + column_index * tile_width
            path, caption = sources[column]
            if column == "clean":
                caption = "Clean"
            panel = exact_square_at(path, 720).resize(
                (tile_width, image_height), Image.Resampling.LANCZOS
            )
            canvas.paste(panel, (x0, y0))
            draw.rectangle(
                (x0, y0 + image_height, x0 + tile_width, y0 + row_height),
                fill="#eef3f7", outline="#b8c1ca", width=2,
            )
            centered(
                draw, (x0, y0 + image_height, x0 + tile_width, y0 + row_height),
                caption, size=36, bold=column != "clean",
                fill="#173f8f" if column != "clean" else "#394a5a",
            )

    label_y = header_height + row_height * len(ROWS)
    draw.rectangle((0, label_y, canvas_width, canvas_height), fill="white")
    for column_index, panel_label in enumerate(("(a)", "(b)", "(c)", "(d)", "(e)")):
        x0 = label_width + column_index * tile_width
        panel_font = serif_font(52)
        bound = draw.textbbox((0, 0), panel_label, font=panel_font)
        draw.text(
            (x0 + (tile_width - (bound[2] - bound[0])) // 2,
             label_y + (panel_height - (bound[3] - bound[1])) // 2 - bound[1]),
            panel_label, font=panel_font, fill="#111111",
        )

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "selected_5pairs_5col_epoch004_factor_tokens_preservation.png"
    canvas.save(destination, optimize=True, dpi=(300, 300))
    return destination


def _svg_centered_text(parent: ET.Element, x0: int, y0: int, x1: int, y1: int,
                       value: str, size: int, *, bold: bool = False,
                       fill: str = "#172433") -> None:
    lines = value.split("\n")
    line_height = int(size * 1.02)
    first_y = (y0 + y1) / 2 - (len(lines) - 1) * line_height / 2
    text = ET.SubElement(parent, "text", {
        "x": str((x0 + x1) / 2), "y": str(first_y),
        "text-anchor": "middle", "dominant-baseline": "middle",
        "font-family": "Times New Roman, Times, STIXGeneral, serif",
        "font-size": str(size), "font-weight": "700" if bold else "400",
        "fill": fill,
    })
    for index, line in enumerate(lines):
        span = ET.SubElement(text, "tspan", {
            "x": str((x0 + x1) / 2),
            "y": str(first_y + index * line_height),
        })
        span.text = line


def _jpeg_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_landscape_4x3_svg() -> Path:
    """Same preserved image panels as the PNG, with resolution-independent text."""
    columns = ("clean", "order1", "order2", "ours")
    headers = {
        "clean": "Clean\n(BSDS500)", "order1": "GenDeg\nOrder 1",
        "order2": "GenDeg\nOrder 2", "ours": "Ours\n(Joint 1-Pass)",
    }
    width, height = 3200, 2400
    label_width, header_height, panel_height = 360, 120, 90
    tile_width = (width - label_width) // 4
    row_height = (height - header_height - panel_height) // 5
    caption_height = 62
    image_height = row_height - caption_height

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "width": str(width),
        "height": str(height), "viewBox": f"0 0 {width} {height}",
    })
    ET.SubElement(svg, "rect", {"width": str(width), "height": str(height), "fill": "white"})
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0", "width": str(label_width), "height": str(header_height),
        "fill": "#dbe7f2", "stroke": "#aab7c2", "stroke-width": "2",
    })
    _svg_centered_text(svg, 0, 0, label_width, header_height, "Degradation", 42, bold=True)
    for ci, column in enumerate(columns):
        x0 = label_width + ci * tile_width
        ET.SubElement(svg, "rect", {
            "x": str(x0), "y": "0", "width": str(tile_width), "height": str(header_height),
            "fill": "#dbe7f2", "stroke": "#aab7c2", "stroke-width": "2",
        })
        _svg_centered_text(svg, x0, 0, x0 + tile_width, header_height,
                           headers[column], 40, bold=True)

    for ri, row in enumerate(ROWS):
        y0 = header_height + ri * row_height
        bg = "#ffffff" if ri % 2 == 0 else "#edf3f8"
        ET.SubElement(svg, "rect", {
            "x": "0", "y": str(y0), "width": str(width), "height": str(row_height),
            "fill": bg, "stroke": "#b8c1ca", "stroke-width": "2",
        })
        _svg_centered_text(svg, 0, y0, label_width, y0 + row_height,
                           row.label.replace(" + ", " +\n"), 45, bold=True)
        sources = row_sources(row)
        for ci, column in enumerate(columns):
            x0 = label_width + ci * tile_width
            path, caption = sources[column]
            if column == "clean":
                caption = "Clean"
            panel = exact_square_at(path, 720)
            ET.SubElement(svg, "image", {
                "x": str(x0), "y": str(y0), "width": str(tile_width),
                "height": str(image_height), "preserveAspectRatio": "none",
                "href": _jpeg_data_uri(panel),
            })
            ET.SubElement(svg, "rect", {
                "x": str(x0), "y": str(y0 + image_height), "width": str(tile_width),
                "height": str(caption_height), "fill": "#eef3f7",
                "stroke": "#b8c1ca", "stroke-width": "2",
            })
            _svg_centered_text(
                svg, x0, y0 + image_height, x0 + tile_width, y0 + row_height,
                caption, 36, bold=column != "clean",
                fill="#173f8f" if column != "clean" else "#394a5a",
            )

    label_y = header_height + row_height * len(ROWS)
    for ci, label in enumerate(("(a)", "(b)", "(c)", "(d)")):
        x0 = label_width + ci * tile_width
        _svg_centered_text(svg, x0, label_y, x0 + tile_width, height, label, 52)

    destination = OUT / "selected_5pairs_4col_landscape_4x3_vector_text.svg"
    ET.ElementTree(svg).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def main() -> None:
    without = build(False)
    with_onerestore = build(True)
    landscape = build_landscape_4x3()
    landscape_epoch4 = build_landscape_epoch4_5col()
    landscape_svg = build_landscape_4x3_svg()
    readme = """# Selected BSDS500 two-factor comparison

- Rows: Haze+Snow (41029), Low-light+Haze (123057), Low-light+Snow (285022), Low-light+Rain (16068), Haze+Rain (290035).
- Every displayed image is center-cropped and resized to the same 360x360 cell.
- GenDeg Order 1 follows the factor order in the row label; Order 2 reverses it.
- Ours uses the epoch-30 simultaneous joint-condition output.
- The added epoch-4 column uses the continuation checkpoint with appended
  per-degradation factor tokens, factor-null contrast loss, and the weighted
  physical/preservation prior. Spatial conditioning is disabled in this checkpoint.
- The epoch-4 inference reuses donor 02134, fixed seed 2031, Euler ancestral
  sampling with 100 steps, and text/image CFG 1.0/1.0 from the existing Ours column.
- OneRestore uses the released CDD-11 physical synthesis recipe. Official ordering is low -> rain/snow -> haze.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(without)
    print(with_onerestore)
    print(landscape)
    print(landscape_epoch4)
    print(landscape_svg)


if __name__ == "__main__":
    main()
