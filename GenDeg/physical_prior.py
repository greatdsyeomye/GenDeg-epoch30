"""Differentiable factor-aware physical priors for full-latent GenDeg.

The losses operate on aligned clean/degraded RGB pairs in [0, 1].  They are
used only as auxiliary supervision at low diffusion timesteps; the diffusion
state remains the complete degraded latent rather than a residual latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PhysicalPriorWeights:
    low_light: float = 1.0
    haze: float = 1.0
    rain: float = 2.0
    snow: float = 1.0
    rain_snow_exclusion: float = 0.0
    structure: float = 0.0


def _luminance(image: Tensor) -> Tensor:
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def _blur(image: Tensor, kernel_size: int) -> Tensor:
    return F.avg_pool2d(
        image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
    )


def _masked_mean(per_sample: Tensor, active: Tensor) -> Tensor:
    active = active.to(dtype=per_sample.dtype)
    return (per_sample * active).sum() / active.sum().clamp_min(1.0)


def _atmospheric_light(target: Tensor) -> Tensor:
    """Estimate RGB atmospheric light from the brightest 1% target pixels."""
    with torch.no_grad():
        luminance = _luminance(target).flatten(2)
        count = max(1, luminance.shape[-1] // 100)
        indices = luminance.topk(count, dim=-1).indices
        rgb = target.flatten(2)
        indices = indices.expand(-1, rgb.shape[1], -1)
        return rgb.gather(2, indices).mean(dim=-1).view(-1, 3, 1, 1)


def _transmission(image: Tensor, clean: Tensor, atmosphere: Tensor, floor: float) -> Tensor:
    """Least-squares transmission for I = J*t + A*(1-t), smoothed spatially."""
    clean_offset = clean - atmosphere
    image_offset = image - atmosphere
    numerator = (image_offset * clean_offset).sum(dim=1, keepdim=True)
    denominator = clean_offset.square().sum(dim=1, keepdim=True).clamp_min(1e-4)
    transmission = (numerator / denominator).clamp(floor, 1.0)
    return _blur(transmission, 15)


def _streak_response(image: Tensor, clean: Tensor) -> Tensor:
    residual = _luminance(image) - _luminance(clean)
    # Bright, thin structures survive removal of the local low-frequency field.
    high_pass = residual - _blur(residual, 9)
    # A hard ReLU has zero gradient when the current prediction contains no
    # rain at all.  The zero-centred softplus keeps a creation gradient while
    # remaining a close approximation to a positive streak detector.
    beta = 40.0
    return F.softplus(high_pass, beta=beta) - high_pass.new_tensor(2.0).log() / beta


def _particle_response(image: Tensor, clean: Tensor) -> Tensor:
    residual = _luminance(image) - _luminance(clean)
    beta = 40.0
    positive = F.softplus(residual, beta=beta) - residual.new_tensor(2.0).log() / beta
    high_pass = positive - _blur(positive, 15)
    return F.softplus(high_pass, beta=beta) - high_pass.new_tensor(2.0).log() / beta


def _normalized_structure_gradients(image: Tensor) -> tuple[Tensor, Tensor]:
    """Return illumination-invariant gradients of the scene-scale luminance field.

    Blurring before differentiation prevents rain streaks and snow particles from
    being treated as content edges. Per-image RMS normalization preserves edge
    geometry without forcing haze/low-light outputs to retain clean contrast.
    """
    base = _blur(_luminance(image), 7)
    grad_x = F.pad(base[:, :, :, 1:] - base[:, :, :, :-1], (0, 1, 0, 0))
    grad_y = F.pad(base[:, :, 1:, :] - base[:, :, :-1, :], (0, 0, 0, 1))
    rms = (grad_x.square() + grad_y.square()).mean((1, 2, 3), keepdim=True).sqrt()
    rms = rms.clamp_min(1e-3)
    return grad_x / rms, grad_y / rms


def composite_physical_prior_loss(
    generated: Tensor,
    clean: Tensor,
    target: Tensor,
    active: Tensor,
    *,
    weights: PhysicalPriorWeights = PhysicalPriorWeights(),
    visibility_floor: float = 0.05,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return balanced low/haze/rain/snow priors and detached diagnostics.

    ``active`` follows the canonical order: low-light, haze, rain, snow.
    Each factor is averaged independently so a frequent or high-amplitude
    degradation cannot numerically erase the rain term.
    """
    if active.ndim != 2 or active.shape[1] != 4:
        raise ValueError(f"active must have shape [B, 4], got {tuple(active.shape)}")

    generated = generated.float().clamp(0.0, 1.0)
    clean = clean.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    clean_y = _luminance(clean)
    generated_y = _luminance(generated)
    target_y = _luminance(target)

    # Low-light: match the spatial attenuation field and prevent black collapse.
    denominator = _blur(clean_y, 15) + 0.02
    generated_attenuation = (_blur(generated_y, 15) / denominator).clamp(0.0, 2.0)
    target_attenuation = (_blur(target_y, 15) / denominator).clamp(0.0, 2.0)
    low_map = (generated_attenuation - target_attenuation).abs().mean((1, 2, 3))
    visibility = F.relu(visibility_floor * clean_y - generated_y).mean((1, 2, 3))
    low_loss = _masked_mean(low_map + visibility, active[:, 0])

    # Haze: enforce spatial transmission and the atmospheric scattering equation.
    atmosphere = _atmospheric_light(target)
    target_t = _transmission(target, clean, atmosphere, visibility_floor).detach()
    generated_t = _transmission(generated, clean, atmosphere, visibility_floor)
    transmission_error = (generated_t - target_t).abs().mean((1, 2, 3))
    reconstructed = clean * generated_t + atmosphere * (1.0 - generated_t)
    scattering_error = (generated - reconstructed).abs().mean((1, 2, 3))
    haze_loss = _masked_mean(transmission_error + 0.25 * scattering_error, active[:, 1])

    # Rain: explicitly preserve visible thin streak energy and its spatial coverage.
    generated_streak = _streak_response(generated, clean)
    target_streak = _streak_response(target, clean).detach()
    streak_map_error = (generated_streak - target_streak).abs().mean((1, 2, 3))
    generated_coverage = torch.sigmoid((generated_streak - 0.015) / 0.005).mean((1, 2, 3))
    target_coverage = torch.sigmoid((target_streak - 0.015) / 0.005).mean((1, 2, 3))
    coverage_error = (generated_coverage - target_coverage).abs()
    rain_loss = _masked_mean(streak_map_error + coverage_error, active[:, 2])

    # Snow: match bright particle residuals without encouraging a global white veil.
    generated_particle = _particle_response(generated, clean)
    target_particle = _particle_response(target, clean).detach()
    particle_map_error = (generated_particle - target_particle).abs().mean((1, 2, 3))
    particle_energy_error = (
        generated_particle.mean((1, 2, 3)) - target_particle.mean((1, 2, 3))
    ).abs()
    snow_loss = _masked_mean(particle_map_error + particle_energy_error, active[:, 3])

    # Rain-only composites must not gain snow-like bright particles beyond what
    # is present in their aligned rain target. This remains disabled for genuine
    # rain+snow samples, where snow particles are desired.
    rain_without_snow = active[:, 2] * (1.0 - active[:, 3])
    particle_excess = F.relu(generated_particle - target_particle - 0.002)
    rain_snow_exclusion_loss = _masked_mean(
        particle_excess.mean((1, 2, 3)), rain_without_snow
    )

    # Preserve scene layout while allowing photometric changes and high-frequency
    # weather. The comparison is against clean scene edges, not degraded texture.
    generated_gx, generated_gy = _normalized_structure_gradients(generated)
    clean_gx, clean_gy = _normalized_structure_gradients(clean)
    structure_error = (
        (generated_gx - clean_gx).abs().mean((1, 2, 3))
        + (generated_gy - clean_gy).abs().mean((1, 2, 3))
    )
    structure_loss = _masked_mean(structure_error, active.amax(dim=1))

    total = (
        weights.low_light * low_loss
        + weights.haze * haze_loss
        + weights.rain * rain_loss
        + weights.snow * snow_loss
        + weights.rain_snow_exclusion * rain_snow_exclusion_loss
        + weights.structure * structure_loss
    )
    diagnostics = {
        "low": low_loss.detach(),
        "haze": haze_loss.detach(),
        "rain": rain_loss.detach(),
        "snow": snow_loss.detach(),
        "rain_snow_exclusion": rain_snow_exclusion_loss.detach(),
        "structure": structure_loss.detach(),
        "rain_generated_coverage": _masked_mean(generated_coverage, active[:, 2]).detach(),
        "rain_target_coverage": _masked_mean(target_coverage, active[:, 2]).detach(),
    }
    return total, diagnostics
