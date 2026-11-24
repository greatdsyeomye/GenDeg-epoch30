"""Datasets for factorized composite-degradation diffusion training."""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from composite_condition import (
    RAIN_SUBTYPE_TO_ID,
    CompositeConditionConfig,
)


CDD_TO_CANONICAL = {
    "low": "low_light",
    "haze": "haze",
    "rain": "rain",
    "snow": "snow",
}
CDD11_CATEGORIES = (
    "low",
    "haze",
    "rain",
    "snow",
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
    "low_haze_rain",
    "low_haze_snow",
)

FACTOR_ALIASES = {
    "low": "low_light",
    "low-light": "low_light",
    "low_light": "low_light",
    "haze": "haze",
    "rain": "rain",
    "snow": "snow",
}


def _instantiate_dataset(config: dict[str, Any]) -> Dataset:
    target = config["target"]
    module_name, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**dict(config.get("params", {})))


def _load_rgb_with_geometry(
    path: Path,
    resolution: int,
    resize_mode: str = "letterbox",
) -> tuple[Tensor, dict[str, Any]]:
    """Load RGB without changing geometry.

    Letterbox mode fits the long side to ``resolution`` and edge-pads the
    short side.  Metadata is returned so inference can remove the padding and
    add the generated residual to the untouched native-resolution image.
    """
    image = Image.open(path).convert("RGB")
    original_width, original_height = image.size
    if resize_mode == "stretch":
        resized_width = resized_height = resolution
        padding = (0, 0, 0, 0)
        crop = (0, 0, resolution, resolution)
    elif resize_mode == "center_crop":
        # Preserve geometry: resize the short side to the requested size and
        # take an aligned square crop.  This avoids both anisotropic stretch
        # and the large repeated borders produced by letterboxing 3:2 data.
        scale = resolution / min(original_width, original_height)
        resized_width = max(resolution, round(original_width * scale))
        resized_height = max(resolution, round(original_height * scale))
        left_crop = (resized_width - resolution) // 2
        top_crop = (resized_height - resolution) // 2
        crop = (
            left_crop,
            top_crop,
            left_crop + resolution,
            top_crop + resolution,
        )
        padding = (0, 0, 0, 0)
    elif resize_mode == "letterbox":
        scale = resolution / max(original_width, original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        pad_width = resolution - resized_width
        pad_height = resolution - resized_height
        padding = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        crop = (0, 0, resolution, resolution)
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    image = image.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    if resize_mode == "center_crop":
        image = image.crop(crop)
    array = np.asarray(image, dtype=np.float32) / 255.0
    left, top, right, bottom = padding
    if any(padding):
        array = np.pad(
            array,
            ((top, bottom), (left, right), (0, 0)),
            mode="edge",
        )
    tensor = rearrange(torch.from_numpy(array.copy()), "h w c -> c h w")
    geometry = {
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "padding": padding,
        "crop": crop,
        "resize_mode": resize_mode,
    }
    return tensor, geometry


def _load_rgb(
    path: Path, resolution: int, resize_mode: str = "letterbox"
) -> Tensor:
    return _load_rgb_with_geometry(path, resolution, resize_mode)[0]


def _rain_subtype_from_text(value: str) -> str:
    value = value.lower()
    has_drop = "raindrop" in value or "rain_drop" in value or "/drop" in value
    has_streak = "streak" in value or ("rain" in value and not has_drop)
    if has_drop and has_streak:
        return "mixed"
    if has_drop:
        return "raindrop"
    if has_streak:
        return "streak"
    return "unspecified"


def _difference_stats(clean: Tensor, degraded: Tensor) -> Tensor:
    difference = (clean - degraded).abs()
    return torch.stack((difference.mean(), difference.std(unbiased=False)))


def _luminance(image: Tensor) -> Tensor:
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return (image * weights).sum(dim=0)


def _box_blur(image: Tensor, kernel_size: int = 31) -> Tensor:
    return F.avg_pool2d(
        image[None, None],
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )[0, 0]


def _otsu_mask(values: Tensor) -> np.ndarray:
    array = values.detach().cpu().numpy()
    uint8 = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(
        uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask > 0


def _dominant_line_orientation(values: Tensor, mask: np.ndarray) -> float:
    """Return an undirected line angle normalized to [0, 1)."""
    array = values.detach().cpu().numpy().astype(np.float32)
    grad_x = cv2.Sobel(array, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(array, cv2.CV_32F, 0, 1, ksize=3)
    if not mask.any():
        return 0.0
    gx = grad_x[mask]
    gy = grad_y[mask]
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    gradient_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    line_angle = (gradient_angle + np.pi / 2.0) % np.pi
    return float(line_angle / np.pi)


def _median_component_diameter(mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    diameters = 2.0 * np.sqrt(areas / np.pi)
    return float(np.clip(np.median(diameters) / max(mask.shape), 0.0, 1.0))


def _observable_factors(
    degradation: str,
    clean: Tensor,
    degraded: Tensor,
) -> Tensor:
    """Measure two non-physical, reproducible factors from an aligned pair."""
    eps = 1e-6
    clean_y = _luminance(clean)
    degraded_y = _luminance(degraded)
    residual = (degraded - clean).abs().mean(dim=0)

    if degradation == "low":
        brightness_ratio = degraded_y.mean() / (clean_y.mean() + eps)
        brightness_loss = (1.0 - brightness_ratio).clamp(0.0, 1.0)
        illumination_ratio = (degraded_y / (clean_y + 0.02)).clamp(0.0, 2.0)
        illumination = _box_blur(illumination_ratio)
        nonuniformity = (
            illumination.std(unbiased=False) / (illumination.mean() + eps)
        ).clamp(0.0, 1.0)
        return torch.stack((brightness_loss, nonuniformity))

    if degradation == "haze":
        contrast_ratio = degraded_y.std(unbiased=False) / (
            clean_y.std(unbiased=False) + eps
        )
        contrast_loss = (1.0 - contrast_ratio).clamp(0.0, 1.0)
        spatial_variation = _box_blur(residual).std(unbiased=False).clamp(0.0, 1.0)
        return torch.stack((contrast_loss, spatial_variation))

    if degradation == "rain":
        mask = _otsu_mask(residual)
        coverage = residual.new_tensor(float(mask.mean()))
        orientation = residual.new_tensor(_dominant_line_orientation(residual, mask))
        return torch.stack((coverage, orientation))

    if degradation == "snow":
        positive_residual = (degraded_y - clean_y).clamp_min(0.0)
        mask = _otsu_mask(positive_residual)
        coverage = residual.new_tensor(float(mask.mean()))
        particle_size = residual.new_tensor(_median_component_diameter(mask))
        return torch.stack((coverage, particle_size))

    raise KeyError(f"Unsupported degradation for observable factors: {degradation}")


def _one_hot(value: Tensor, maximum: float, bins: int = 128) -> Tensor:
    index = int(torch.clamp(value / maximum * bins, 0, bins - 1).item())
    encoded = torch.zeros(bins + 1)
    encoded[index] = 1
    return encoded


def _severity_word(mu: Tensor) -> str:
    """Temporary CDD-11 severity labels; replace with train-split quantiles."""
    value = float(mu)
    if value < 0.08:
        return "light"
    if value < 0.18:
        return "moderate"
    return "heavy"


class CDD11CompositeDataset(Dataset):
    """Return GenDeg and factorized conditions from aligned CDD-11 folders.

    CDD-11 publishes type labels but not per-image physical parameters. This
    loader therefore measures observable factors from aligned single variants.
    They are component proxies, not an exact decomposition of the composite.
    Folder-name order is unknown by default and is not treated as supervision.
    """

    def __init__(
        self,
        root: str | Path,
        resolution: int = 256,
        categories: Sequence[str] = CDD11_CATEGORIES,
        max_clean_images: int | None = None,
        assume_category_order: bool = False,
        resize_mode: str = "letterbox",
        identity_probability: float = 0.0,
        prompt_mode: str = "factorized",
        flip_probability: float = 0.0,
        **_: Any,
    ) -> None:
        self.root = Path(root)
        self.resolution = resolution
        self.categories = tuple(categories)
        self.assume_category_order = assume_category_order
        self.resize_mode = resize_mode
        self.identity_probability = identity_probability
        if prompt_mode not in {"factorized", "atomic", "generic"}:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        self.prompt_mode = prompt_mode
        self.flip_probability = flip_probability
        self.factor_config = CompositeConditionConfig()
        self.factor_names = list(self.factor_config.degradation_types)

        clean_files = sorted((self.root / "clear").glob("*.png"))
        if max_clean_images is not None:
            clean_files = clean_files[:max_clean_images]
        if not clean_files:
            raise FileNotFoundError(f"No clean PNG images found under {self.root / 'clear'}")

        self.samples: list[tuple[Path, Path, str]] = []
        for clean_path in clean_files:
            for category in self.categories:
                target_path = self.root / category / clean_path.name
                if target_path.is_file():
                    self.samples.append((clean_path, target_path, category))
        if not self.samples:
            raise FileNotFoundError(f"No aligned degraded images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clean_path, target_path, category = self.samples[index]
        clean = _load_rgb(clean_path, self.resolution, self.resize_mode)
        target = _load_rgb(target_path, self.resolution, self.resize_mode)
        use_flip = random.random() < self.flip_probability
        if use_flip:
            clean = torch.flip(clean, dims=(-1,))
            target = torch.flip(target, dims=(-1,))

        use_identity = random.random() < self.identity_probability
        if use_identity:
            target = clean.clone()

        global_mu, global_sigma = _difference_stats(clean, target)
        global_stats = torch.stack(
            (_one_hot(global_mu, 0.7), _one_hot(global_sigma, 0.4))
        )

        k = len(self.factor_names)
        p = self.factor_config.parameter_dim
        factors = {
            "active": torch.zeros(k),
            "stats": torch.zeros(k, 2),
            "stats_mask": torch.zeros(k),
            "parameters": torch.zeros(k, p),
            "parameter_mask": torch.zeros(k, p),
            "order": torch.full((k,), -1, dtype=torch.long),
            "subtype": torch.zeros(k, dtype=torch.long),
        }

        active_cdd_names = [] if use_identity else category.split("_")
        prompt_descriptors = []
        for stage, cdd_name in enumerate(active_cdd_names):
            canonical_name = CDD_TO_CANONICAL[cdd_name]
            factor_index = self.factor_names.index(canonical_name)
            single_path = self.root / cdd_name / clean_path.name
            if not single_path.is_file():
                raise FileNotFoundError(
                    f"Aligned single-degradation image is required: {single_path}"
                )
            single = _load_rgb(single_path, self.resolution, self.resize_mode)
            if use_flip:
                single = torch.flip(single, dims=(-1,))
            component_stats = _difference_stats(clean, single)
            observable_factors = _observable_factors(cdd_name, clean, single)
            factors["active"][factor_index] = 1
            factors["stats"][factor_index] = component_stats
            factors["stats_mask"][factor_index] = 1
            factors["parameters"][factor_index] = observable_factors
            factors["parameter_mask"][factor_index] = 1
            if self.assume_category_order:
                factors["order"][factor_index] = stage
            if canonical_name == "rain":
                factors["subtype"][factor_index] = RAIN_SUBTYPE_TO_ID["streak"]
            readable_name = cdd_name.replace("low", "low-light")
            prompt_descriptors.append(
                f"{_severity_word(component_stats[0])} {readable_name}"
            )

        if use_identity:
            prompt = "a clean outdoor scene without degradation"
        elif self.prompt_mode == "generic":
            # Removes the degradation-name shortcut so the factor branch must
            # carry type and intensity information during controlled training.
            prompt = "an outdoor scene"
        elif self.prompt_mode == "atomic":
            # B1 treats the complete folder label as one category.  It does
            # not expose component-wise severities or interaction structure.
            readable_category = category.replace("low", "low-light").replace("_", " ")
            prompt = f"an outdoor scene with {readable_category} degradation"
        else:
            prompt = "an outdoor scene with " + " and ".join(prompt_descriptors)

        return {
            "edited": target * 2 - 1,
            "edit": {
                "c_concat": clean * 2 - 1,
                "c_crossattn": prompt,
                "c_stats": global_stats,
                "c_factors": factors,
            },
            "category": "identity" if use_identity else category,
            "image_id": clean_path.name,
            "dataset": "cdd11",
        }


class ManifestCompositeDataset(Dataset):
    """Common paired-data contract for VCD, CCDD-11, RFIR, BID, and others.

    Each JSONL row minimally contains ``clean``, ``target``, and ``active``.
    Paths are relative to ``root`` unless absolute. Optional factor labels use::

        {"factors": {"haze": {"stats": [mu, sigma],
         "parameters": [p0, p1], "parameter_mask": [1, 1], "order": 0}}}

    Missing physical labels remain masked instead of being fabricated.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path | None = None,
        resolution: int = 256,
        dataset_name: str = "manifest",
        resize_mode: str = "letterbox",
        identity_probability: float = 0.0,
        flip_probability: float = 0.0,
        prompt_mode: str = "record",
        **_: Any,
    ) -> None:
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.parent
        self.resolution = resolution
        self.dataset_name = dataset_name
        self.resize_mode = resize_mode
        self.identity_probability = identity_probability
        if prompt_mode not in {"record", "generic"}:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0, 1]")
        self.flip_probability = flip_probability
        with self.manifest.open("r", encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        self.factor_config = CompositeConditionConfig()
        self.factor_names = list(self.factor_config.degradation_types)

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        clean_path = self._path(record["clean"])
        target_path = self._path(record["target"])
        clean = _load_rgb(clean_path, self.resolution, self.resize_mode)
        target = _load_rgb(target_path, self.resolution, self.resize_mode)
        if random.random() < self.flip_probability:
            clean = torch.flip(clean, dims=(-1,))
            target = torch.flip(target, dims=(-1,))
        use_identity = random.random() < self.identity_probability
        if use_identity:
            target = clean.clone()
        global_mu, global_sigma = _difference_stats(clean, target)
        global_stats = torch.stack(
            (_one_hot(global_mu, 0.7), _one_hot(global_sigma, 0.4))
        )

        k = len(self.factor_names)
        p = self.factor_config.parameter_dim
        factors = {
            "active": torch.zeros(k),
            "stats": torch.zeros(k, 2),
            "stats_mask": torch.zeros(k),
            "parameters": torch.zeros(k, p),
            "parameter_mask": torch.zeros(k, p),
            "order": torch.full((k,), -1, dtype=torch.long),
            "subtype": torch.zeros(k, dtype=torch.long),
        }
        active_names = (
            [] if use_identity else [FACTOR_ALIASES[name] for name in record["active"]]
        )
        labels = record.get("factors", {})
        for name in active_names:
            factor_index = self.factor_names.index(name)
            factors["active"][factor_index] = 1
            label = labels.get(name, labels.get(name.replace("low_light", "low"), {}))
            if "stats" in label:
                factors["stats"][factor_index] = torch.as_tensor(label["stats"])
                factors["stats_mask"][factor_index] = 1
            if "parameters" in label:
                values = torch.as_tensor(label["parameters"], dtype=torch.float32)
                factors["parameters"][factor_index, : len(values)] = values[:p]
                mask = label.get("parameter_mask", [1] * len(values))
                mask = torch.as_tensor(mask, dtype=torch.float32)
                factors["parameter_mask"][factor_index, : len(mask)] = mask[:p]
            if "order" in label:
                factors["order"][factor_index] = int(label["order"])
            if name == "rain":
                subtype = label.get("subtype", record.get("rain_subtype"))
                if subtype is None:
                    subtype = _rain_subtype_from_text(
                        f"{target_path.as_posix()} {record.get('category', '')}"
                    )
                factors["subtype"][factor_index] = RAIN_SUBTYPE_TO_ID.get(
                    str(subtype), RAIN_SUBTYPE_TO_ID["unspecified"]
                )

        category = "identity" if use_identity else record.get(
            "category", "_".join(active_names)
        )
        prompt = record.get(
            "prompt",
            "an outdoor scene with "
            + " and ".join(name.replace("_", "-") for name in active_names),
        )
        if use_identity:
            prompt = "a clean scene without degradation"
        elif self.prompt_mode == "generic":
            prompt = "an outdoor scene"
        return {
            "edited": target * 2 - 1,
            "edit": {
                "c_concat": clean * 2 - 1,
                "c_crossattn": prompt,
                "c_stats": global_stats,
                "c_factors": factors,
            },
            "category": category,
            "image_id": record.get("image_id", clean_path.name),
            "dataset": self.dataset_name,
        }


class CompositeDatasetMixture(Dataset):
    """Deterministically sample multiple composite datasets by weight."""

    def __init__(
        self,
        datasets: Sequence[dict[str, Any]],
        weights: Sequence[float] | None = None,
        epoch_size: int | None = None,
        seed: int = 23,
        sampling: str = "weighted",
        prompt_mode: str | None = None,
        **_: Any,
    ) -> None:
        dataset_configs = []
        for config in datasets:
            copied = dict(config)
            copied["params"] = dict(config.get("params", {}))
            if prompt_mode is not None:
                copied["params"]["prompt_mode"] = prompt_mode
            dataset_configs.append(copied)
        self.datasets = [_instantiate_dataset(config) for config in dataset_configs]
        if not self.datasets:
            raise ValueError("At least one dataset config is required")
        self.weights = list(weights or [1.0] * len(self.datasets))
        if len(self.weights) != len(self.datasets) or sum(self.weights) <= 0:
            raise ValueError("weights must match datasets and have a positive sum")
        total = float(sum(self.weights))
        self.weights = [weight / total for weight in self.weights]
        self.epoch_size = epoch_size or sum(len(dataset) for dataset in self.datasets)
        self.seed = seed
        self.sampling = sampling
        if sampling not in {"weighted", "concat"}:
            raise ValueError("sampling must be 'weighted' or 'concat'")
        if sampling == "concat":
            self.epoch_size = sum(len(dataset) for dataset in self.datasets)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.sampling == "concat":
            offset = index % self.epoch_size
            for dataset in self.datasets:
                if offset < len(dataset):
                    return dataset[offset]
                offset -= len(dataset)
            raise IndexError(index)
        generator = random.Random(self.seed + index)
        dataset_index = generator.choices(
            range(len(self.datasets)), weights=self.weights, k=1
        )[0]
        dataset = self.datasets[dataset_index]
        sample_index = generator.randrange(len(dataset))
        return dataset[sample_index]
