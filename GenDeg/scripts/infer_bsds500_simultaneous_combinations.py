#!/usr/bin/env python3
"""BSDS500 zero-shot inference for simultaneous composite conditions.

Unlike sequential GenDeg inference, every factor combination is applied once
to the original clean image.  Factor order is intentionally left unset (-1),
so combinations, not permutations, define the experiment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import einops
import k_diffusion as K
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "stable_diffusion"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "taming-transformers"))
os.chdir(PROJECT_ROOT)

from composite_condition import make_null_composite_condition  # noqa: E402
from composite_dataset import (  # noqa: E402
    CDD11CompositeDataset,
    _load_rgb_with_geometry,
    _one_hot,
    _severity_word,
)
from ldm.util import instantiate_from_config  # noqa: E402


FACTORS = ("low", "haze", "rain", "snow")
CANONICAL = {"low": "low_light", "haze": "haze", "rain": "rain", "snow": "snow"}
DISPLAY = {"low": "low-light", "haze": "haze", "rain": "rain", "snow": "snow"}
SUPPORTED_CDD11 = {
    "low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow",
    "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
}
DEFAULT_IMAGE_IDS = ("123057", "198087", "285022", "290035", "41029", "118031", "16068")


class CompositeCFGDenoiser(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (3 b) ...")
        cfg_sigma = einops.repeat(sigma, "b ... -> (3 b) ...")
        cfg_cond = {
            "c_crossattn": [torch.cat((cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]))],
            "c_concat": [torch.cat((cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]))],
        }
        if "c_factor_crossattn" in cond:
            cfg_cond["c_factor_crossattn"] = [torch.cat((
                cond["c_factor_crossattn"][0],
                uncond["c_factor_crossattn"][0],
                uncond["c_factor_crossattn"][0],
            ))]
            cfg_cond["c_factor_mask"] = torch.cat((
                cond["c_factor_mask"],
                uncond["c_factor_mask"],
                uncond["c_factor_mask"],
            ))
        out_cond, out_image, out_uncond = self.inner_model(
            cfg_z, cfg_sigma, cond=cfg_cond
        ).chunk(3)
        return (
            out_uncond
            + text_cfg_scale * (out_cond - out_image)
            + image_cfg_scale * (out_image - out_uncond)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, default=Path("data/cdd11/train"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-text", type=float, default=1.0)
    parser.add_argument("--cfg-image", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--prompt-mode",
        choices=("factorized", "atomic", "generic"),
        default="factorized",
        help="Prompt mode used for CDD-11 conditions; match the training config.",
    )
    parser.add_argument(
        "--fixed-noise-seed",
        action="store_true",
        help="Use --seed for every input image instead of adding the image index.",
    )
    parser.add_argument("--max-order", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument(
        "--only-combinations",
        nargs="+",
        default=None,
        help="Optional exact categories, e.g. low_haze low_rain haze_rain.",
    )
    parser.add_argument("--image-ids", nargs="+", default=list(DEFAULT_IMAGE_IDS))
    parser.add_argument(
        "--image-combination-map",
        nargs="+",
        default=None,
        metavar="IMAGE_ID=CATEGORY",
        help=(
            "Generate only one requested combination for each listed image, "
            "e.g. 41029=haze_snow 123057=low_haze. This is useful for "
            "matched qualitative tables without rendering the full Cartesian product."
        ),
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Process every supported image in --clean-root instead of --image-ids.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip an image when every requested generated output already exists.",
    )
    parser.add_argument(
        "--no-grids",
        action="store_true",
        help="Do not create per-image or dataset comparison grids (recommended for bulk generation).",
    )
    parser.add_argument(
        "--donor-id",
        default=None,
        help="Use one explicit aligned CDD-11 donor (for controlled ablations).",
    )
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--spatial-mode",
        choices=("neither", "weather", "flare", "both"),
        default="both",
        help="Ablate illumination-aware weather and light-flare branches.",
    )
    parser.add_argument(
        "--factor-stat",
        action="append",
        default=[],
        metavar="FACTOR:MU:SIGMA",
        help=(
            "Override one factor's scalar statistics while keeping the global "
            "condition fixed, e.g. rain:0.08:0.04. May be repeated."
        ),
    )
    parser.add_argument(
        "--global-stat",
        default=None,
        metavar="MU:SIGMA",
        help="Optionally override the composite-level global statistics.",
    )
    parser.add_argument(
        "--factor-token-intervention",
        choices=("normal", "null", "zero", "scale", "swap"),
        default="normal",
        help=(
            "Causal intervention on the factor-token branch. null replaces the "
            "conditional tokens by the encoded null condition; zero removes the "
            "whole branch from both conditional and unconditional contexts; scale "
            "multiplies only the active-condition delta from the null tokens; swap "
            "cyclically moves factor controls to the wrong degradation slots."
        ),
    )
    parser.add_argument(
        "--factor-token-scale",
        type=float,
        default=1.0,
        help="Active factor-token delta multiplier used with --factor-token-intervention scale.",
    )
    return parser.parse_args()


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().float().clamp(-1, 1).permute(1, 2, 0) + 1.0) * 127.5)
    return Image.fromarray(array.round().byte().cpu().numpy())


def add_factor_tokens(model, cross_attention, factor_tokens):
    if factor_tokens is None:
        return cross_attention
    if model.composite_condition_injection in {"additive", "pooled_residual"}:
        return cross_attention + factor_tokens.mean(dim=1, keepdim=True)
    return torch.cat((cross_attention, factor_tokens), dim=1)


def swap_factor_controls(factors):
    """Move each factor control to the next slot without changing other conditions."""
    swapped = {}
    for name, value in factors.items():
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == 4:
            swapped[name] = torch.roll(value, shifts=1, dims=1)
        else:
            swapped[name] = value
    return swapped


def intervene_factor_tokens(tokens, null_tokens, mode, scale):
    if tokens is None:
        return None
    if mode == "normal":
        return tokens
    if mode == "null":
        return null_tokens
    if mode == "zero":
        return torch.zeros_like(tokens)
    if mode == "scale":
        return null_tokens + scale * (tokens - null_tokens)
    raise ValueError(f"Token intervention {mode!r} must be handled before encoding")


def add_factor_residual_condition(model, condition, factor_tokens, factors):
    if factor_tokens is None or model.composite_condition_injection != "pooled_residual":
        return
    active = factors["active"].bool()
    condition["c_factor_crossattn"] = [factor_tokens * active.unsqueeze(-1)]
    condition["c_factor_mask"] = active.to(factor_tokens)


def build_cross_attention(model, text_tokens, stats, factor_tokens):
    """Build either legacy GenDeg context or explicit global/factor tokens."""
    if model.global_stats_injection == "append":
        zero_features = torch.zeros(
            text_tokens.shape[0], text_tokens.shape[1], 2,
            device=text_tokens.device, dtype=text_tokens.dtype,
        )
        cross_attention = model.cc_projection(
            torch.cat((text_tokens, zero_features), dim=-1)
        )
        global_tokens = model.global_stats_token_encoder(stats)
        cross_attention = torch.cat((cross_attention, global_tokens), dim=1)
    else:
        stats_tokens = model.stats_proj(stats).transpose(1, 2).contiguous()
        cross_attention = model.cc_projection(
            torch.cat((text_tokens, stats_tokens), dim=-1)
        )
    cross_attention = add_factor_tokens(model, cross_attention, factor_tokens)
    if (
        model.global_stats_injection == "append"
        and model.composite_condition_injection == "append"
        and tuple(cross_attention.shape[1:]) != (83, 768)
    ):
        raise RuntimeError(
            f"Expected [B, 83, 768] inference context, got {tuple(cross_attention.shape)}"
        )
    return cross_attention


def labelled_tile(image: Image.Image, label: str, size: int = 256) -> Image.Image:
    tile = Image.new("RGB", (size, size + 26), "white")
    preview = image.copy()
    preview.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile.paste(preview, ((size - preview.width) // 2, 26 + (size - preview.height) // 2))
    ImageDraw.Draw(tile).text((5, 6), label, fill="black")
    return tile


def save_grid(items: list[tuple[str, Image.Image]], path: Path, columns: int) -> None:
    tile_size = 256
    rows = math.ceil(len(items) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + 26)), "white")
    for index, (label, image) in enumerate(items):
        canvas.paste(
            labelled_tile(image, label, tile_size),
            ((index % columns) * tile_size, (index // columns) * (tile_size + 26)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def category_name(combination: tuple[str, ...]) -> str:
    return "_".join(combination)


def make_synthesized_condition(
    combination: tuple[str, ...], single_samples: dict[str, dict]
) -> dict:
    """Compose an order-free condition for a CDD-11-unobserved combination."""
    template = single_samples[combination[0]]["edit"]["c_factors"]
    factors = {
        "active": torch.zeros_like(template["active"]),
        "stats": torch.zeros_like(template["stats"]),
        "stats_mask": torch.zeros_like(template["stats_mask"]),
        "parameters": torch.zeros_like(template["parameters"]),
        "parameter_mask": torch.zeros_like(template["parameter_mask"]),
        "order": torch.full_like(template["order"], -1),
        "subtype": torch.zeros_like(template["subtype"]),
    }
    descriptors = []
    component_mus = []
    component_sigmas = []
    factor_names = ("low_light", "haze", "rain", "snow")
    for factor in combination:
        index = factor_names.index(CANONICAL[factor])
        source = single_samples[factor]["edit"]["c_factors"]
        for key in ("active", "stats", "stats_mask", "parameters", "parameter_mask", "subtype"):
            factors[key][index] = source[key][index]
        # No stage/order supervision: every active factor retains order=-1.
        mu = float(factors["stats"][index, 0])
        sigma = float(factors["stats"][index, 1])
        component_mus.append(mu)
        component_sigmas.append(sigma)
        descriptors.append(f"{_severity_word(torch.tensor(mu))} {DISPLAY[factor]}")

    # Missing CDD-11 categories have no measured composite target.  Approximate
    # its global control conservatively while retaining exact per-factor stats.
    global_mu = min(0.7, sum(component_mus))
    global_sigma = min(0.4, math.sqrt(sum(value * value for value in component_sigmas)))
    global_stats = torch.stack(
        (_one_hot(torch.tensor(global_mu), 0.7), _one_hot(torch.tensor(global_sigma), 0.4))
    )
    return {
        "c_crossattn": "an outdoor scene with " + " and ".join(descriptors),
        "c_stats": global_stats,
        "c_factors": factors,
        "condition_source": "synthesized_from_aligned_single_factors",
        "synthesized_global_mu_sigma": [global_mu, global_sigma],
    }


def main() -> None:
    args = parse_args()
    factor_overrides = {}
    factor_aliases = {**CANONICAL, **{name: name for name in CANONICAL.values()}}
    for specification in args.factor_stat:
        try:
            factor_name, mu, sigma = specification.split(":", 2)
            factor_name = factor_aliases[factor_name]
            factor_overrides[factor_name] = (float(mu), float(sigma))
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid --factor-stat {specification!r}; expected "
                "low|low_light|haze|rain|snow:MU:SIGMA"
            ) from error
    global_override = None
    if args.global_stat is not None:
        try:
            global_override = tuple(float(value) for value in args.global_stat.split(":"))
        except ValueError as error:
            raise ValueError("--global-stat must be MU:SIGMA") from error
        if len(global_override) != 2:
            raise ValueError("--global-stat must be MU:SIGMA")
    image_combination_map = None
    if args.image_combination_map:
        image_combination_map = {}
        for item in args.image_combination_map:
            try:
                image_id, category = item.split("=", 1)
            except ValueError as error:
                raise ValueError(
                    f"Invalid --image-combination-map item {item!r}; expected IMAGE_ID=CATEGORY"
                ) from error
            if image_id in image_combination_map:
                raise ValueError(f"Duplicate mapped image ID: {image_id}")
            image_combination_map[image_id] = category
        args.image_ids = list(image_combination_map)
    if args.all_images:
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        clean_files = sorted(
            path for path in args.clean_root.iterdir()
            if path.is_file() and path.suffix.lower() in supported
        )
        if not clean_files:
            raise FileNotFoundError(f"No images found under {args.clean_root}")
        args.image_ids = [path.stem for path in clean_files]
    elif not args.image_ids:
        raise ValueError("At least one image ID is required")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_paths = []
    for image_id in args.image_ids:
        matches = [
            path for suffix in (".jpg", ".jpeg", ".png")
            if (path := args.clean_root / f"{image_id}{suffix}").is_file()
        ]
        if not matches:
            raise FileNotFoundError(f"BSDS500 image not found: {image_id}")
        clean_paths.append(matches[0])

    combinations_by_order = {
        order: list(itertools.combinations(FACTORS, order))
        for order in range(1, args.max_order + 1)
    }
    if args.only_combinations:
        requested = set(args.only_combinations)
        known = {
            category_name(combo)
            for combos in combinations_by_order.values()
            for combo in combos
        }
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown or out-of-range combinations: {sorted(unknown)}")
        combinations_by_order = {
            order: [combo for combo in combos if category_name(combo) in requested]
            for order, combos in combinations_by_order.items()
            if any(category_name(combo) in requested for combo in combos)
        }
    categories = [category_name(combo) for combos in combinations_by_order.values() for combo in combos]
    if image_combination_map is not None:
        unknown_mapped = set(image_combination_map.values()) - set(categories)
        if unknown_mapped:
            raise ValueError(
                f"Mapped combinations are not enabled by the requested order/combination set: "
                f"{sorted(unknown_mapped)}"
            )
    # Single-factor donors are also needed to synthesize any combination that
    # is absent from CDD-11, even when only pair outputs were requested.
    dataset_categories = sorted(set(categories) | set(FACTORS))
    datasets = {
        category: CDD11CompositeDataset(
            args.donor_root,
            resolution=args.resolution,
            categories=(category,),
            resize_mode="center_crop",
            prompt_mode=args.prompt_mode,
        )
        for category in sorted(set(dataset_categories) & SUPPORTED_CDD11)
    }
    sample_indices = {
        category: {clean_path.name: index for index, (clean_path, _, _) in enumerate(dataset.samples)}
        for category, dataset in datasets.items()
    }
    donor_ids = sorted(path.name for path in (args.donor_root / "clear").glob("*.png"))
    donor_rng = random.Random(args.seed)
    if args.donor_id is not None:
        donor_id = args.donor_id if args.donor_id.endswith(".png") else f"{args.donor_id}.png"
        if donor_id not in donor_ids:
            raise FileNotFoundError(f"CDD-11 donor not found: {donor_id}")
        selected_donors = {image_id: donor_id for image_id in args.image_ids}
    else:
        selected_donors = {image_id: donor_rng.choice(donor_ids) for image_id in args.image_ids}

    config = OmegaConf.load(args.project_config)
    config.model.params.ckpt_path = None
    if config.model.params.get("use_ema", False):
        config.model.params.load_ema = True
    model = instantiate_from_config(config.model)
    payload = torch.load(args.ckpt, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    if hasattr(model.cond_stage_model, "device"):
        model.cond_stage_model.device = str(device)

    denoiser = K.external.CompVisDenoiser(model)
    guided_denoiser = CompositeCFGDenoiser(denoiser)
    sigmas = denoiser.get_sigmas(args.steps)
    null_stats = torch.zeros(1, 2, 129, device=device)
    null_stats[:, :, -1] = 1.0
    ema_context = nullcontext() if args.no_ema else model.ema_scope()

    report = {
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch_one_based": int(payload.get("epoch", -1)) + 1,
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "project_config": str(args.project_config),
        "device": args.device,
        "resolution": args.resolution,
        "sampler": "Euler ancestral",
        "steps": args.steps,
        "cfg_text": args.cfg_text,
        "cfg_image": args.cfg_image,
        "prompt_mode": args.prompt_mode,
        "base_seed": args.seed,
        "seed_policy": (
            "same fixed base_seed for every image and combination"
            if args.fixed_noise_seed
            else "same base_seed + image_index for every combination"
        ),
        "factors": list(FACTORS),
        "simultaneous_application": True,
        "sequential_application": False,
        "latent_target": "residual" if model.residual_diffusion else "full",
        "spatial_conditioning": model.spatial_condition_encoder is not None,
        "spatial_mode": args.spatial_mode,
        "global_mu_sigma_disabled": bool(model.disable_global_stats),
        "factor_mu_sigma_disabled": bool(model.disable_factor_stats),
        "global_stats_injection": model.global_stats_injection,
        "factor_stats_injection": model.composite_condition_injection,
        "decode_mode": (
            "clean_plus_decoded_residual"
            if model.residual_diffusion
            else "decode_sampled_full_latent"
        ),
        "order_condition": -1,
        "unsupported_cdd11_combinations": [
            category for category in categories if category not in SUPPORTED_CDD11
        ],
        "samples": [],
        "image_combination_map": image_combination_map,
        "factor_stat_overrides": factor_overrides,
        "global_stat_override": global_override,
        "factor_token_intervention": args.factor_token_intervention,
        "factor_token_scale": args.factor_token_scale,
    }
    if model.spatial_condition_encoder is not None:
        spatial_parameters = torch.cat(
            [parameter.detach().float().flatten().cpu()
             for parameter in model.spatial_condition_encoder.parameters()]
        )
        report["spatial_parameter_abs_mean"] = float(spatial_parameters.abs().mean())
        report["spatial_parameter_abs_max"] = float(spatial_parameters.abs().max())
    all_outputs = {
        combination: []
        for combinations in combinations_by_order.values()
        for combination in combinations
    }
    clean_images = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16), ema_context:
        null_text = model.get_learned_conditioning([""])
        if model.composite_condition_encoder is not None:
            null_factors = make_null_composite_condition(
                1, model.composite_condition_encoder.config, device=device
            )
            null_factor_tokens = model.composite_condition_encoder(**null_factors)
        else:
            null_factors = None
            null_factor_tokens = None
        unconditional_factor_tokens = (
            torch.zeros_like(null_factor_tokens)
            if null_factor_tokens is not None and args.factor_token_intervention == "zero"
            else null_factor_tokens
        )
        null_cross_attention = build_cross_attention(
            model, null_text, null_stats, unconditional_factor_tokens
        )
        null_condition = {
            "c_crossattn": [null_cross_attention],
            "c_concat": None,
        }
        add_factor_residual_condition(
            model, null_condition, null_factor_tokens, null_factors
        )
        report["cross_attention_shape"] = list(null_cross_attention.shape)

        for image_index, (image_id, clean_path) in enumerate(zip(args.image_ids, clean_paths)):
            clean_01, geometry = _load_rgb_with_geometry(clean_path, args.resolution, "center_crop")
            clean = clean_01 * 2 - 1
            clean_batch = clean.unsqueeze(0).to(device)
            clean_latent_mode = model.encode_first_stage(clean_batch).mode()
            clean_latent_scaled = model.scale_factor * clean_latent_mode
            clean_reconstruction = model.decode_first_stage(clean_latent_scaled)
            clean_image = to_image(clean)
            clean_images.append(clean_image)
            (args.output_dir / "clean_native").mkdir(exist_ok=True)
            (args.output_dir / "clean_model").mkdir(exist_ok=True)
            Image.open(clean_path).convert("RGB").save(args.output_dir / "clean_native" / f"{image_id}.png")
            clean_image.save(args.output_dir / "clean_model" / f"{image_id}.png")

            donor_id = selected_donors[image_id]
            single_samples = {
                factor: datasets[factor][sample_indices[factor][donor_id]] for factor in FACTORS
            }
            outputs_for_image = {}
            condition_records = {}
            requested_for_image = image_combination_map.get(image_id) if image_combination_map else None
            expected_outputs = [
                args.output_dir / f"order{len(combination)}" / category_name(combination) / f"{image_id}.png"
                for combinations in combinations_by_order.values()
                for combination in combinations
                if requested_for_image is None or category_name(combination) == requested_for_image
            ]
            if args.resume and all(path.is_file() for path in expected_outputs):
                print(
                    f"Resume: skipped complete image {image_index + 1}/{len(args.image_ids)}: {image_id}",
                    flush=True,
                )
                continue
            for order, combinations in combinations_by_order.items():
                for combination in combinations:
                    category = category_name(combination)
                    if requested_for_image is not None and category != requested_for_image:
                        continue
                    if category in datasets:
                        donor_sample = datasets[category][sample_indices[category][donor_id]]
                        edit = donor_sample["edit"]
                        condition_source = "measured_aligned_cdd11_combination"
                        synthesized_global = None
                    else:
                        synthesized = make_synthesized_condition(combination, single_samples)
                        edit = synthesized
                        condition_source = synthesized["condition_source"]
                        synthesized_global = synthesized["synthesized_global_mu_sigma"]

                    stats = edit["c_stats"].unsqueeze(0).to(device)
                    if global_override is not None:
                        global_mu, global_sigma = global_override
                        stats = torch.stack((
                            _one_hot(torch.tensor(global_mu), 0.7),
                            _one_hot(torch.tensor(global_sigma), 0.4),
                        )).unsqueeze(0).to(device)
                    stats = model.prepare_global_stats(stats)
                    factors = {
                        name: value.unsqueeze(0).to(device)
                        for name, value in edit["c_factors"].items()
                    }
                    factors = model.prepare_factor_condition(factors)
                    factor_names = ("low_light", "haze", "rain", "snow")
                    for factor_name, (mu, sigma) in factor_overrides.items():
                        factor_index = factor_names.index(factor_name)
                        factors["stats"][:, factor_index, 0] = mu
                        factors["stats"][:, factor_index, 1] = sigma
                        factors["stats_mask"][:, factor_index] = True
                    if torch.any(factors["order"] != -1):
                        raise RuntimeError(f"Order leaked into condition for {category}")
                    text_tokens = model.get_learned_conditioning([edit["c_crossattn"]])
                    factor_encoder_inputs = (
                        swap_factor_controls(factors)
                        if args.factor_token_intervention == "swap"
                        else factors
                    )
                    factor_tokens = (
                        model.composite_condition_encoder(**factor_encoder_inputs)
                        if model.composite_condition_encoder is not None
                        else None
                    )
                    if args.factor_token_intervention != "swap":
                        factor_tokens = intervene_factor_tokens(
                            factor_tokens,
                            null_factor_tokens,
                            args.factor_token_intervention,
                            args.factor_token_scale,
                        )
                    cross_attention = build_cross_attention(
                        model, text_tokens, stats, factor_tokens
                    )
                    spatial_clean_condition = model.build_spatial_condition(
                        clean_batch, clean_latent_mode, factors,
                        spatial_mode=args.spatial_mode,
                    )
                    spatial_delta = (
                        spatial_clean_condition[:, clean_latent_mode.shape[1]:]
                        if spatial_clean_condition.shape[1] > clean_latent_mode.shape[1]
                        else spatial_clean_condition - clean_latent_mode
                    )
                    cond = {
                        "c_crossattn": [cross_attention],
                        "c_concat": [spatial_clean_condition],
                    }
                    add_factor_residual_condition(
                        model, cond, factor_tokens, factors
                    )
                    uncond = {
                        "c_crossattn": [null_cross_attention],
                        "c_concat": [torch.zeros_like(spatial_clean_condition)],
                    }
                    if model.composite_condition_injection == "pooled_residual":
                        uncond["c_factor_crossattn"] = null_condition[
                            "c_factor_crossattn"
                        ]
                        uncond["c_factor_mask"] = null_condition["c_factor_mask"]
                    sample_seed = args.seed if args.fixed_noise_seed else args.seed + image_index
                    torch.manual_seed(sample_seed)
                    latent = torch.randn_like(clean_latent_mode) * sigmas[0]
                    latent = K.sampling.sample_euler_ancestral(
                        guided_denoiser,
                        latent,
                        sigmas,
                        extra_args={
                            "cond": cond,
                            "uncond": uncond,
                            "text_cfg_scale": args.cfg_text,
                            "image_cfg_scale": args.cfg_image,
                        },
                        disable=True,
                    )
                    if model.residual_diffusion:
                        degraded_reconstruction = model.decode_first_stage(
                            clean_latent_scaled + latent
                        )
                        generated = (
                            clean_batch
                            + degraded_reconstruction
                            - clean_reconstruction
                        )[0].clamp(-1, 1)
                    else:
                        # A full-latent model samples the degraded image latent
                        # directly. Adding the clean latent again would decode
                        # z_clean + z_degraded, which does not match training.
                        generated = model.decode_first_stage(latent)[0].clamp(-1, 1)
                    generated_image = to_image(generated)
                    outputs_for_image[combination] = generated_image
                    all_outputs[combination].append(generated_image)
                    output_dir = args.output_dir / f"order{order}" / category
                    output_dir.mkdir(parents=True, exist_ok=True)
                    generated_image.save(output_dir / f"{image_id}.png")
                    condition_records[category] = {
                        "source": condition_source,
                        "prompt": edit["c_crossattn"],
                        "synthesized_global_mu_sigma": synthesized_global,
                        "active": factors["active"][0].detach().cpu().tolist(),
                        "order": factors["order"][0].detach().cpu().tolist(),
                        "spatial_delta_abs_mean": float(spatial_delta.float().abs().mean()),
                        "spatial_delta_abs_max": float(spatial_delta.float().abs().max()),
                        "factor_token_abs_mean": (
                            float(factor_tokens.float().abs().mean())
                            if factor_tokens is not None else 0.0
                        ),
                        "factor_token_delta_from_null_abs_mean": (
                            float((factor_tokens - null_factor_tokens).float().abs().mean())
                            if factor_tokens is not None and null_factor_tokens is not None else 0.0
                        ),
                    }

            if not args.no_grids:
                compare_dir = args.output_dir / "comparisons" / image_id
                for order, combinations in combinations_by_order.items():
                    items = [("clean", clean_image)] + [
                        (" + ".join(DISPLAY[name] for name in combo), outputs_for_image[combo])
                        for combo in combinations
                    ]
                    columns = 3 if order == 2 else 5
                    save_grid(items, compare_dir / f"order{order}_all.jpg", columns)
            report["samples"].append(
                {
                    "image_id": image_id,
                    "clean_path": str(clean_path),
                    "original_size": geometry["original_size"],
                    "donor_image_id": donor_id,
                    "noise_seed": args.seed if args.fixed_noise_seed else args.seed + image_index,
                    "conditions": condition_records,
                }
            )
            print(f"Completed {image_index + 1}/{len(args.image_ids)}: {image_id}", flush=True)

    if not args.no_grids:
        for combination, images in all_outputs.items():
            category = category_name(combination)
            items = []
            for image_id, clean, generated in zip(args.image_ids, clean_images, images):
                items.extend(((f"{image_id} clean", clean), (category, generated)))
            save_grid(
                items,
                args.output_dir / f"order{len(combination)}" / category / "comparison_all.jpg",
                2,
            )

    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    total = sum(len(images) for images in all_outputs.values())
    print(f"Saved {total} simultaneous-combination images under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
