#!/usr/bin/env python3
"""Quantitative BSDS500 comparison of sequential GenDeg and joint generation.

The BSDS500 images have no paired degraded targets.  Consequently, content
metrics measure preservation relative to each method's own preprocessed clean
input, while degradation proxies measure visible changes rather than fidelity.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from composite_dataset import _observable_factors


VANILLA = ROOT / "results/bsds500_vanilla_gendeg_ordered_60ep_seed2026"
OURS = ROOT / "results/20261019_112259_KST_bsds500_gendeg_init_no_llr_512_epoch030_cfg1x1_7img"
OUT = ROOT / "results/bsds500_gendeg_vs_ours_quantitative_7img"
IMAGE_IDS = ("123057", "198087", "285022", "290035", "41029", "118031", "16068")
FACTORS = ("low-light", "haze", "rain", "snow")
OURS_NAMES = {"low-light": "low", "haze": "haze", "rain": "rain", "snow": "snow"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    with Image.open(path) as image:
        result = image.convert("RGB")
    if size is not None and result.size != size:
        result = result.resize(size, Image.Resampling.LANCZOS)
    return result


def image_tensor(image: Image.Image) -> torch.Tensor:
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value.copy()).permute(2, 0, 1)


def edge_correlation(clean: np.ndarray, generated: np.ndarray) -> float:
    def magnitude(value: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(dx * dx + dy * dy).reshape(-1)

    first, second = magnitude(clean), magnitude(generated)
    if first.std() < 1e-8 or second.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def paths(first: str, second: str, image_id: str) -> dict[str, tuple[Path, Path]]:
    vanilla_clean = VANILLA / "clean" / f"{image_id}.png"
    ours_clean = OURS / "clean_model" / f"{image_id}.png"
    vanilla = VANILLA / "generated/order2" / f"{first}__to__{second}" / f"{image_id}.png"
    names = tuple(sorted((OURS_NAMES[first], OURS_NAMES[second]), key=("low", "haze", "rain", "snow").index))
    ours = OURS / "order2" / "_".join(names) / f"{image_id}.png"
    return {"gendeg_2pass": (vanilla_clean, vanilla), "ours_1pass": (ours_clean, ours)}


def load_dino(device: torch.device):
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(device)


def dino_feature_map(paths_to_images: dict[Path, Image.Image], device: torch.device,
                     batch_size: int) -> dict[Path, np.ndarray]:
    model = load_dino(device)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    items = list(paths_to_images.items())
    result: dict[Path, np.ndarray] = {}
    with torch.inference_mode():
        for offset in range(0, len(items), batch_size):
            subset = items[offset:offset + batch_size]
            batch = []
            for _, image in subset:
                value = F.interpolate(
                    image_tensor(image).unsqueeze(0), size=(224, 224),
                    mode="bicubic", align_corners=False,
                )[0]
                batch.append((value - mean) / std)
            features = F.normalize(model(torch.stack(batch).to(device)).float(), dim=-1).cpu().numpy()
            for (path, _), feature in zip(subset, features):
                result[path] = feature
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bulk_median_seconds(root: Path) -> float:
    timestamps = sorted(path.stat().st_mtime for path in root.rglob("*.png"))
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:]) if 0.1 < b - a < 60.0]
    if not intervals:
        raise RuntimeError(f"No usable generation intervals under {root}")
    return float(statistics.median(intervals))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    records = []
    images: dict[Path, Image.Image] = {}
    for first, second in itertools.permutations(FACTORS, 2):
        sequence = f"{first}__to__{second}"
        for image_id in IMAGE_IDS:
            for method, (clean_path, output_path) in paths(first, second, image_id).items():
                if not clean_path.is_file() or not output_path.is_file():
                    raise FileNotFoundError(clean_path if not clean_path.is_file() else output_path)
                clean = load_rgb(clean_path)
                generated = load_rgb(output_path, clean.size)
                images.setdefault(clean_path, clean)
                # The same joint output is intentionally shared by both orders.
                images.setdefault(output_path, generated)
                clean_array = np.asarray(clean, dtype=np.uint8)
                generated_array = np.asarray(generated, dtype=np.uint8)
                clean_tensor = image_tensor(clean)
                generated_tensor = image_tensor(generated)
                low = _observable_factors("low", clean_tensor, generated_tensor)
                haze = _observable_factors("haze", clean_tensor, generated_tensor)
                rain = _observable_factors("rain", clean_tensor, generated_tensor)
                snow = _observable_factors("snow", clean_tensor, generated_tensor)
                records.append({
                    "image_id": image_id,
                    "sequence": sequence,
                    "method": method,
                    "clean_path": str(clean_path),
                    "output_path": str(output_path),
                    "edge_correlation": edge_correlation(clean_array, generated_array),
                    "psnr_to_clean_db": float(peak_signal_noise_ratio(clean_array, generated_array, data_range=255)),
                    "ssim_to_clean": float(structural_similarity(clean_array, generated_array, data_range=255, channel_axis=-1)),
                    "brightness_loss": float(low[0]),
                    "haze_contrast_loss": float(haze[0]),
                    "rain_artifact_coverage": float(rain[0]),
                    "snow_artifact_coverage": float(snow[0]),
                })

    features = dino_feature_map(images, device, args.batch_size)
    for row in records:
        clean_feature = features[Path(row["clean_path"])]
        output_feature = features[Path(row["output_path"])]
        row["dino_cosine_similarity"] = float(clean_feature @ output_feature)
        del row["clean_path"], row["output_path"]

    metric_names = (
        "edge_correlation", "dino_cosine_similarity", "psnr_to_clean_db", "ssim_to_clean",
        "brightness_loss", "haze_contrast_loss", "rain_artifact_coverage", "snow_artifact_coverage",
    )
    sequence_rows = []
    for sequence in sorted({row["sequence"] for row in records}):
        for method in ("gendeg_2pass", "ours_1pass"):
            selected = [row for row in records if row["sequence"] == sequence and row["method"] == method]
            output = {"sequence": sequence, "method": method, "n": len(selected)}
            output.update({name: float(np.mean([row[name] for row in selected])) for name in metric_names})
            sequence_rows.append(output)

    overall_rows = []
    for method in ("gendeg_2pass", "ours_1pass"):
        selected = [row for row in records if row["method"] == method]
        output = {"method": method, "n": len(selected)}
        output.update({name: float(np.mean([row[name] for row in selected])) for name in metric_names})
        overall_rows.append(output)

    factor_metrics = {
        "low-light": "brightness_loss",
        "haze": "haze_contrast_loss",
        "rain": "rain_artifact_coverage",
        "snow": "snow_artifact_coverage",
    }
    factor_rows = []
    for factor, metric_name in factor_metrics.items():
        for method in ("gendeg_2pass", "ours_1pass"):
            selected = [
                row for row in records
                if row["method"] == method and factor in row["sequence"].split("__to__")
            ]
            factor_rows.append({
                "active_factor": factor,
                "visibility_proxy": metric_name,
                "method": method,
                "n": len(selected),
                "mean_value": float(np.mean([row[metric_name] for row in selected])),
            })

    vanilla_pass = bulk_median_seconds(
        ROOT / "data/restoration_transfer_pilot/generated/vanilla_gendeg/test/generated"
    )
    ours_pass = bulk_median_seconds(
        ROOT / "data/restoration_transfer_pilot/generated/epoch030/test/order2"
    )
    efficiency = [
        {"method": "GenDeg sequential 2-pass", "diffusion_passes": 2, "denoising_steps": 200,
         "median_seconds_per_pass": vanilla_pass, "estimated_pair_seconds": 2 * vanilla_pass,
         "relative_compute": 2.0, "speedup_vs_sequential": 1.0},
        {"method": "Ours joint 1-pass", "diffusion_passes": 1, "denoising_steps": 100,
         "median_seconds_per_pass": ours_pass, "estimated_pair_seconds": ours_pass,
         "relative_compute": 1.0, "speedup_vs_sequential": 2 * vanilla_pass / ours_pass},
    ]

    write_csv(args.output_dir / "quantitative_per_image.csv", records)
    write_csv(args.output_dir / "quantitative_by_sequence.csv", sequence_rows)
    write_csv(args.output_dir / "quantitative_overall.csv", overall_rows)
    write_csv(args.output_dir / "degradation_by_active_factor.csv", factor_rows)
    write_csv(args.output_dir / "inference_efficiency.csv", efficiency)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"overall": overall_rows, "efficiency": efficiency}, indent=2), encoding="utf-8"
    )

    lookup = {row["method"]: row for row in overall_rows}
    lines = [
        "# BSDS500 quantitative evaluation: sequential GenDeg vs. joint generation", "",
        "BSDS500 has no paired degraded ground truth. Edge/DINO/PSNR/SSIM therefore measure source preservation, not degradation realism. The four degradation columns are reproducible residual-based visibility proxies.", "",
        "| Method | Edge Corr. ↑ | DINO Cos. ↑ | PSNR ↑ | SSIM ↑ | Brightness loss | Haze contrast loss | Rain coverage | Snow coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, label in (("gendeg_2pass", "GenDeg sequential 2-pass"), ("ours_1pass", "Ours joint 1-pass")):
        row = lookup[method]
        lines.append(
            f"| {label} | {row['edge_correlation']:.4f} | {row['dino_cosine_similarity']:.4f} | "
            f"{row['psnr_to_clean_db']:.2f} | {row['ssim_to_clean']:.4f} | {row['brightness_loss']:.4f} | "
            f"{row['haze_contrast_loss']:.4f} | {row['rain_artifact_coverage']:.4f} | {row['snow_artifact_coverage']:.4f} |"
        )
    lines.extend(["", "| Method | Passes | Denoising steps | Time / pair ↓ | Relative compute ↓ | Speedup ↑ |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in efficiency:
        lines.append(
            f"| {row['method']} | {row['diffusion_passes']} | {row['denoising_steps']} | "
            f"{row['estimated_pair_seconds']:.2f} s | {row['relative_compute']:.1f}× | {row['speedup_vs_sequential']:.2f}× |"
        )
    lines.extend(["", "| Active factor | Proxy | GenDeg 2-pass | Ours 1-pass |",
                  "|---|---|---:|---:|"])
    factor_lookup = {(row["active_factor"], row["method"]): row for row in factor_rows}
    for factor, metric_name in factor_metrics.items():
        baseline = factor_lookup[(factor, "gendeg_2pass")]["mean_value"]
        proposed = factor_lookup[(factor, "ours_1pass")]["mean_value"]
        lines.append(f"| {factor} | {metric_name} | {baseline:.4f} | {proposed:.4f} |")
    lines.extend(["", "Higher visibility-proxy values indicate a stronger measured change, not necessarily better realism. In particular, they should ultimately be compared with the corresponding real-degradation distribution.", "",
                  "Timing is estimated from the median interval between outputs in the completed 200-image bulk runs on GPU 0; model loading is excluded and image-save overhead is included."])
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "README.md")


if __name__ == "__main__":
    main()
