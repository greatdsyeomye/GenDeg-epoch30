import json

import numpy as np
import torch
from PIL import Image

from composite_dataset import (
    CDD11CompositeDataset,
    CompositeDatasetMixture,
    ManifestCompositeDataset,
    _load_rgb_with_geometry,
)


def test_cdd11_aligned_haze_rain_sample():
    dataset = CDD11CompositeDataset(
        "data/cdd11/smoke/test",
        categories=("haze_rain",),
        max_clean_images=1,
    )
    sample = dataset[0]
    factors = sample["edit"]["c_factors"]

    assert sample["edited"].shape == (3, 256, 256)
    assert sample["edit"]["c_stats"].shape == (2, 129)
    assert factors["active"].tolist() == [0.0, 1.0, 1.0, 0.0]
    # CDD category names describe a set, not a verified application order.
    assert factors["order"].tolist() == [-1, -1, -1, -1]
    assert factors["stats"][1:3].gt(0).all()
    assert factors["stats_mask"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert factors["parameter_mask"][1:3].eq(1).all()
    assert factors["parameters"][1:3].isfinite().all()
    assert factors["parameters"][1:3].ge(0).all()
    assert factors["parameters"][1:3].le(1).all()
    assert factors["subtype"].tolist() == [0, 0, 1, 0]
    assert sample["category"] == "haze_rain"
    assert sample["dataset"] == "cdd11"


def test_manifest_dataset_keeps_missing_labels_masked(tmp_path):
    clean = np.full((16, 16, 3), 180, dtype=np.uint8)
    target = np.full((16, 16, 3), 120, dtype=np.uint8)
    Image.fromarray(clean).save(tmp_path / "clean.png")
    Image.fromarray(target).save(tmp_path / "target.png")
    row = {
        "clean": "clean.png",
        "target": "target.png",
        "active": ["low", "haze"],
        "factors": {
            "haze": {
                "parameters": [0.7, 0.2],
                "parameter_mask": [1, 1],
                "order": 1,
            }
        },
    }
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    sample = ManifestCompositeDataset(manifest, dataset_name="vcd")[0]
    factors = sample["edit"]["c_factors"]
    assert factors["active"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert factors["parameter_mask"][0].tolist() == [0.0, 0.0]
    assert factors["stats_mask"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert torch.allclose(factors["parameters"][1], torch.tensor([0.7, 0.2]))
    assert factors["order"].tolist() == [-1, 1, -1, -1]
    assert sample["dataset"] == "vcd"


def test_letterbox_preserves_aspect_ratio(tmp_path):
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "wide.png")
    tensor, geometry = _load_rgb_with_geometry(tmp_path / "wide.png", 64)
    assert tensor.shape == (3, 64, 64)
    assert geometry["resized_size"] == (64, 32)
    assert geometry["padding"] == (0, 16, 0, 16)


def test_center_crop_preserves_geometry_without_padding(tmp_path):
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "wide.png")
    tensor, geometry = _load_rgb_with_geometry(
        tmp_path / "wide.png", 64, resize_mode="center_crop"
    )
    assert tensor.shape == (3, 64, 64)
    assert geometry["resized_size"] == (128, 64)
    assert geometry["crop"] == (32, 0, 96, 64)
    assert geometry["padding"] == (0, 0, 0, 0)


def test_rain_subtype_is_kept_under_parent_factor(tmp_path):
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "clean.png")
    Image.fromarray(image).save(tmp_path / "drop.png")
    manifest = tmp_path / "rain.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clean": "clean.png",
                "target": "drop.png",
                "active": ["rain"],
                "rain_subtype": "raindrop",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    factors = ManifestCompositeDataset(manifest)[0]["edit"]["c_factors"]
    assert factors["active"].tolist() == [0, 0, 1, 0]
    assert factors["subtype"].tolist() == [0, 0, 2, 0]


def test_weighted_mixture_uses_common_contract(tmp_path):
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "a.png")
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps({"clean": "a.png", "target": "a.png", "active": ["rain"]})
        + "\n",
        encoding="utf-8",
    )
    mixture = CompositeDatasetMixture(
        datasets=[
            {
                "target": "composite_dataset.ManifestCompositeDataset",
                "params": {"manifest": str(manifest), "dataset_name": "bid"},
            }
        ],
        epoch_size=3,
    )
    assert len(mixture) == 3
    assert mixture[2]["dataset"] == "bid"
    assert mixture[2]["edit"]["c_factors"]["active"].tolist() == [0, 0, 1, 0]
