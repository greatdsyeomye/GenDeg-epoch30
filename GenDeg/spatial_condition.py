"""Scene-adaptive spatial conditioning for full-latent GenDeg.

The module derives illumination and structure priors only from the clean input
and combines them with an order-free active-factor vector.  It predicts a
small four-channel correction to the existing clean VAE conditioning latent;
the diffusion target itself remains the full degraded latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatialConditionConfig:
    num_factors: int = 4
    hidden_channels: int = 32
    output_channels: int = 4
    max_delta: float = 1.0
    illumination_kernel: int = 31
    mode: str = "latent_delta"
    include_flare: bool = False
    learned_residual_scale: float = 0.25
    max_factor_sum: float = 1.5
    uniform_factor_level: float = 0.65
    # Optional continuous, factor-wise strength control.  The scales are the
    # CDD-11 train-split 95th percentiles of |clean - single degradation|.
    # Missing component statistics fall back to ``uniform_factor_level``;
    # they are never fabricated from a composite target.
    use_factor_strength: bool = False
    factor_mu_scales: Sequence[float] = (0.3095, 0.2462, 0.1551, 0.1250)
    minimum_active_strength: float = 0.05
    # Separate factor experts prevent a shared token/MLP from silently
    # averaging degradation identities.  A zero-init interaction head learns
    # only the correction needed when multiple factors coexist.
    use_factor_experts: bool = False
    interaction_scale: float = 0.25


class SpatialConditionEncoder(nn.Module):
    """Predict a latent spatial condition from scene layout and factor types.

    Input RGB is expected in ``[-1, 1]``.  No degraded/target image statistic
    is used.  The zero-initialized output preserves the old checkpoint exactly
    before fine-tuning while still allowing the final layer to learn at step 1.
    """

    _SCENE_CHANNELS = 8  # RGB + luminance/illumination/dark/highlight/contrast

    def __init__(self, config: SpatialConditionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpatialConditionConfig()
        if self.config.illumination_kernel < 1 or self.config.illumination_kernel % 2 == 0:
            raise ValueError("illumination_kernel must be a positive odd integer")
        if self.config.mode not in {"latent_delta", "factor_maps"}:
            raise ValueError("mode must be 'latent_delta' or 'factor_maps'")
        expected_outputs = self.config.num_factors + int(self.config.include_flare)
        if self.config.mode == "factor_maps" and self.config.output_channels != expected_outputs:
            raise ValueError(
                f"factor_maps mode needs {expected_outputs} output channels, "
                f"got {self.config.output_channels}"
            )
        if len(self.config.factor_mu_scales) != self.config.num_factors:
            raise ValueError("factor_mu_scales must have one value per factor")
        if any(float(value) <= 0 for value in self.config.factor_mu_scales):
            raise ValueError("factor_mu_scales must be positive")

        # Active factors are supplied both directly and multiplied by the
        # illumination map, allowing different weather visibility around dark
        # areas and light sources without imposing a fixed hand-crafted rule.
        input_channels = self._SCENE_CHANNELS + 2 * self.config.num_factors
        hidden = self.config.hidden_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, self.config.output_channels, 3, padding=1),
        )
        nn.init.zeros_(self.backbone[-1].weight)
        nn.init.zeros_(self.backbone[-1].bias)

        if self.config.use_factor_experts:
            # Each expert sees the same clean-scene representation but only
            # its own scalar gate.  This provides an explicit causal path
            # from factor strength to one spatial conditioning channel.
            expert_inputs = self._SCENE_CHANNELS + 2
            expert_hidden = max(8, hidden // 2)
            self.factor_experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(expert_inputs, expert_hidden, 3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(expert_hidden, expert_hidden, 3, padding=1),
                        nn.SiLU(),
                        nn.Conv2d(expert_hidden, 1, 3, padding=1),
                    )
                    for _ in range(self.config.num_factors)
                ]
            )
            for expert in self.factor_experts:
                nn.init.zeros_(expert[-1].weight)
                nn.init.zeros_(expert[-1].bias)

            interaction_inputs = (
                self._SCENE_CHANNELS + 2 * self.config.num_factors
            )
            self.interaction_head = nn.Sequential(
                nn.Conv2d(interaction_inputs, expert_hidden, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(expert_hidden, self.config.num_factors, 3, padding=1),
            )
            nn.init.zeros_(self.interaction_head[-1].weight)
            nn.init.zeros_(self.interaction_head[-1].bias)
        else:
            self.factor_experts = None
            self.interaction_head = None

    def scene_features(self, clean_rgb: Tensor) -> Tensor:
        if clean_rgb.ndim != 4 or clean_rgb.shape[1] != 3:
            raise ValueError(
                f"clean_rgb must have shape [B, 3, H, W], got {tuple(clean_rgb.shape)}"
            )
        rgb = clean_rgb.float().clamp(-1, 1).add(1).mul(0.5)
        weights = rgb.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
        luminance = (rgb * weights).sum(dim=1, keepdim=True)
        kernel = self.config.illumination_kernel
        illumination = F.avg_pool2d(
            luminance, kernel_size=kernel, stride=1, padding=kernel // 2
        )
        darkness = 1.0 - illumination
        highlight = ((illumination - 0.6) / 0.4).clamp(0, 1)
        local_contrast = (luminance - illumination).abs()
        return torch.cat(
            (rgb, luminance, illumination, darkness, highlight, local_contrast), dim=1
        )

    def forward(
        self,
        clean_rgb: Tensor,
        active: Tensor,
        *,
        output_size: tuple[int, int],
        spatial_mode: str = "both",
        stats: Tensor | None = None,
        stats_mask: Tensor | None = None,
    ) -> Tensor:
        if active.ndim != 2 or active.shape != (
            clean_rgb.shape[0], self.config.num_factors
        ):
            raise ValueError(
                "active must have shape "
                f"[{clean_rgb.shape[0]}, {self.config.num_factors}], got {tuple(active.shape)}"
            )
        scene = self.scene_features(clean_rgb)
        scene = F.interpolate(scene, size=output_size, mode="bilinear", align_corners=False)
        active_maps = active.to(device=scene.device, dtype=scene.dtype)[:, :, None, None]
        active_maps = active_maps.expand(-1, -1, *output_size)
        strength = self._factor_strength(active, stats, stats_mask).to(
            device=scene.device, dtype=scene.dtype
        )
        strength_maps = strength[:, :, None, None].expand(-1, -1, *output_size)
        illumination = scene[:, 4:5]
        inputs = torch.cat(
            (scene, strength_maps, strength_maps * illumination), dim=1
        )
        inputs = inputs.to(dtype=self.backbone[0].weight.dtype)
        shared_learned = self.backbone(inputs)
        if self.config.mode == "latent_delta":
            return self.config.max_delta * torch.tanh(shared_learned)

        learned = shared_learned
        if self.factor_experts is not None:
            expert_outputs = []
            for index, expert in enumerate(self.factor_experts):
                gate = strength_maps[:, index : index + 1]
                expert_input = torch.cat((scene, gate, gate * illumination), dim=1)
                expert_outputs.append(expert(expert_input.to(inputs.dtype)))
            factor_learned = torch.cat(expert_outputs, dim=1)
            if self.config.include_flare:
                learned = torch.cat(
                    (factor_learned, shared_learned[:, self.config.num_factors :]),
                    dim=1,
                )
            else:
                learned = factor_learned
        return self._factor_maps(
            scene, active_maps, strength_maps, learned, spatial_mode
        )

    def _factor_strength(
        self,
        active: Tensor,
        stats: Tensor | None,
        stats_mask: Tensor | None,
    ) -> Tensor:
        """Return bounded scalar gates without decomposing composite targets.

        Known per-factor statistics come from aligned single-degradation pairs
        or manifest labels.  Missing labels use a neutral active level so the
        spatial branch remains usable on real/unlabelled composites.
        """
        active = active.float()
        if not self.config.use_factor_strength:
            return active
        fallback = torch.full_like(active, self.config.uniform_factor_level)
        if stats is None:
            return active * fallback
        expected = (*active.shape, 2)
        if tuple(stats.shape) != expected:
            raise ValueError(f"stats must have shape {expected}, got {tuple(stats.shape)}")
        if stats_mask is None:
            stats_mask = torch.ones_like(active)
        elif tuple(stats_mask.shape) != tuple(active.shape):
            raise ValueError(
                f"stats_mask must have shape {tuple(active.shape)}, "
                f"got {tuple(stats_mask.shape)}"
            )
        scales = active.new_tensor(tuple(self.config.factor_mu_scales)).unsqueeze(0)
        normalized = (stats[..., 0].float() / scales).clamp(0.0, 1.0)
        normalized = normalized.clamp_min(self.config.minimum_active_strength)
        strength = torch.where(stats_mask.bool(), normalized, fallback)
        return active * strength

    def _factor_maps(
        self,
        scene: Tensor,
        active_maps: Tensor,
        strength_maps: Tensor,
        learned: Tensor,
        spatial_mode: str,
    ) -> Tensor:
        """Build four bounded weather maps and an optional light-source map.

        The fixed priors make the conditioning spatial from the first update;
        the CNN learns dataset-specific corrections.  Active weather maps are
        jointly capped so adding factors cannot increase their total condition
        without bound.
        """
        if spatial_mode not in {"neither", "weather", "flare", "both"}:
            raise ValueError(
                "spatial_mode must be one of: neither, weather, flare, both"
            )
        illumination = scene[:, 4:5].clamp(0, 1)
        darkness = scene[:, 5:6].clamp(0, 1)
        highlight = scene[:, 6:7].clamp(0, 1)
        contrast = scene[:, 7:8].clamp(0, 1)

        priors = torch.cat(
            (
                0.35 + 0.65 * darkness,
                0.35 + 0.65 * (1.0 - (4.0 * contrast).clamp(0, 1)),
                0.25 + 0.75 * illumination,
                0.20 + 0.80 * illumination.sqrt(),
            ),
            dim=1,
        )
        adaptive = spatial_mode in {"weather", "both"}
        if adaptive:
            factor_maps = priors + self.config.learned_residual_scale * torch.tanh(
                learned[:, : self.config.num_factors]
            )
        else:
            factor_maps = torch.full_like(priors, self.config.uniform_factor_level)
        factor_maps = factor_maps.clamp(0, 1) * strength_maps

        # Interaction is learned only for genuine composites and cannot turn
        # on an inactive factor.  Zero initialization makes the initial model
        # exactly the independent-expert construction.
        if self.interaction_head is not None and adaptive:
            interaction_input = torch.cat((scene, factor_maps, active_maps), dim=1)
            interaction = self.interaction_head(
                interaction_input.to(dtype=learned.dtype)
            )
            composite_mask = (active_maps.sum(dim=1, keepdim=True) > 1).to(factor_maps)
            factor_maps = factor_maps + (
                self.config.interaction_scale
                * torch.tanh(interaction)
                * active_maps
                * composite_mask
            )
            factor_maps = factor_maps.clamp(0, 1)

        total = factor_maps.sum(dim=1, keepdim=True)
        scale = (self.config.max_factor_sum / total.clamp_min(1e-6)).clamp(max=1.0)
        factor_maps = factor_maps * scale

        if not self.config.include_flare:
            return factor_maps
        flare_enabled = spatial_mode in {"flare", "both"}
        weather_active = active_maps[:, 1:].amax(dim=1, keepdim=True)
        if flare_enabled:
            flare = highlight + self.config.learned_residual_scale * torch.tanh(
                learned[:, self.config.num_factors : self.config.num_factors + 1]
            )
            flare = flare.clamp(0, 1) * weather_active
        else:
            flare = torch.zeros_like(highlight)
        return torch.cat((factor_maps, flare), dim=1)
