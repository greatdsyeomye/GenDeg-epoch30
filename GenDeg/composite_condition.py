"""Structured conditioning modules for CompositeGenDeg.

The original GenDeg represents degradation severity with two global tokens
(mu and sigma).  This module keeps the 768-dimensional Stable Diffusion
cross-attention interface, but produces one token for each degradation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


# Two observable controls per degradation. These names intentionally avoid
# claiming unavailable physical ground truth (for example beta or gamma).
# Values are measured from aligned clean/degraded pairs or supplied masks;
# missing values are masked by the dataset loader.
FACTOR_SPECS = {
    "low_light": ("brightness_loss", "illumination_nonuniformity"),
    "haze": ("contrast_loss", "haze_spatial_variation"),
    "rain": ("streak_coverage", "streak_orientation"),
    "snow": ("particle_coverage", "particle_size"),
}

# Rain remains one top-level degradation so that its shared appearance is
# learned jointly.  The subtype disambiguates thin streaks from lens-adherent
# drops without spending additional top-level factor slots.
RAIN_SUBTYPE_TO_ID = {
    "none": 0,
    "streak": 1,
    "raindrop": 2,
    "mixed": 3,
    "unspecified": 4,
}

RAIN_SUBTYPE_FACTOR_SPECS = {
    "streak": ("streak_coverage", "streak_orientation"),
    "raindrop": ("drop_coverage", "drop_size"),
    "mixed": ("rain_coverage", "drop_to_streak_ratio"),
    "unspecified": ("rain_strength", "rain_spatial_variation"),
}


@dataclass(frozen=True)
class CompositeConditionConfig:
    degradation_types: Sequence[str] = ("low_light", "haze", "rain", "snow")
    parameter_dim: int = 2
    hidden_dim: int = 256
    context_dim: int = 768
    num_subtypes: int = len(RAIN_SUBTYPE_TO_ID)
    stats_only: bool = False
    zero_gate: bool = False


@dataclass(frozen=True)
class GlobalStatsTokenConfig:
    """Configuration for the two global (mu, sigma) cross-attention tokens."""

    bins: int = 129
    hidden_dim: int = 256
    context_dim: int = 768
    zero_gate: bool = False


class GlobalStatsTokenEncoder(nn.Module):
    """Project global composite statistics to separate mu/sigma tokens.

    The input is the original GenDeg one-hot tensor ``[B, 2, 129]``.  Unlike
    the legacy path, the statistics are not spread over the 77 CLIP tokens;
    each statistic retains its own cross-attention slot.
    """

    def __init__(self, config: GlobalStatsTokenConfig | None = None) -> None:
        super().__init__()
        self.config = config or GlobalStatsTokenConfig()
        self.stat_type_embedding = nn.Embedding(2, self.config.hidden_dim)
        self.numeric_projection = nn.Sequential(
            nn.Linear(self.config.bins, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.context_dim),
        )
        # 1e-5 underflows the upstream MLP gradients under fp16 autocast.
        # 1e-3 keeps the new branch small while allowing all layers to learn
        # from the first optimization step.
        nn.init.normal_(self.output_projection[-1].weight, std=1e-3)
        nn.init.zeros_(self.output_projection[-1].bias)
        if self.config.zero_gate:
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gate", None)

    def forward(self, stats: Tensor) -> Tensor:
        if stats.ndim != 3 or tuple(stats.shape[1:]) != (2, self.config.bins):
            raise ValueError(
                "global stats must have shape [B, 2, "
                f"{self.config.bins}], got {tuple(stats.shape)}"
            )
        dtype = self.numeric_projection[0].weight.dtype
        stats = stats.to(dtype=dtype)
        type_ids = torch.arange(2, device=stats.device)
        hidden = self.numeric_projection(stats)
        hidden = hidden + self.stat_type_embedding(type_ids).unsqueeze(0)
        tokens = self.output_projection(hidden)
        if self.gate is not None:
            tokens = torch.tanh(self.gate) * tokens
        return tokens


class CompositeConditionEncoder(nn.Module):
    """Encode factorized degradation controls as cross-attention tokens.

    Tensor conventions, where K is the number of degradation types and P is
    ``parameter_dim``:

    * active: [B, K]
    * stats: [B, K, 2] (component-wise mu and sigma)
    * parameters / parameter_mask: [B, K, P]
    * order: [B, K], with -1 for inactive factors

    The observable-factor slots are degradation-specific. Their meanings are
    defined by ``FACTOR_SPECS``; ``parameter_mask`` prevents missing or
    estimator-only values from becoming accidental conditions.
    """

    def __init__(self, config: CompositeConditionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CompositeConditionConfig()
        self.num_degradations = len(self.config.degradation_types)

        self.type_embedding = nn.Embedding(
            self.num_degradations, self.config.hidden_dim
        )
        # 0 is the null/inactive order, and 1..K are actual stages.
        self.order_embedding = nn.Embedding(
            self.num_degradations + 1, self.config.hidden_dim
        )
        self.subtype_embedding = nn.Embedding(
            self.config.num_subtypes, self.config.hidden_dim
        )

        # The paper-scale factor-wise model intentionally uses only factor
        # presence and per-factor (mu, sigma).  The legacy path remains for
        # old checkpoints that also used handcrafted parameters.
        numeric_dim = (
            1 + 2 + 1
            if self.config.stats_only
            else 1 + 2 + 2 * self.config.parameter_dim
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.context_dim),
        )

        # Starting near zero lets an SD/GenDeg backbone initially ignore the
        # new tokens while preserving gradient flow into the whole encoder.
        # Keep the appended tokens initially weak, but not so small that fp16
        # gradients into the statistic projection underflow to zero.
        nn.init.normal_(self.output_projection[-1].weight, std=1e-3)
        nn.init.zeros_(self.output_projection[-1].bias)
        if self.config.zero_gate:
            # At initialization the factor branch is exactly equivalent to
            # vanilla GenDeg.  The scalar gate learns whether the new branch
            # is useful before gradients alter the factor encoder itself.
            self.gate = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("gate", None)

    def forward(
        self,
        *,
        active: Tensor,
        stats: Tensor,
        parameters: Tensor,
        parameter_mask: Tensor,
        order: Tensor,
        subtype: Tensor,
        stats_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_degradations = active.shape
        expected = self.num_degradations
        if num_degradations != expected:
            raise ValueError(
                f"Expected {expected} degradation factors, got {num_degradations}."
            )

        parameter_dim = self.config.parameter_dim
        expected_shapes = {
            "stats": (batch_size, expected, 2),
            "parameters": (batch_size, expected, parameter_dim),
            "parameter_mask": (batch_size, expected, parameter_dim),
            "order": (batch_size, expected),
            "subtype": (batch_size, expected),
        }
        actual = {
            "stats": tuple(stats.shape),
            "parameters": tuple(parameters.shape),
            "parameter_mask": tuple(parameter_mask.shape),
            "order": tuple(order.shape),
            "subtype": tuple(subtype.shape),
        }
        for name, expected_shape in expected_shapes.items():
            if actual[name] != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {actual[name]}."
                )

        dtype = self.numeric_projection[0].weight.dtype
        active = active.to(dtype=dtype)
        stats = stats.to(dtype=dtype)
        parameters = parameters.to(dtype=dtype)
        parameter_mask = parameter_mask.to(dtype=dtype)
        if stats_mask is None:
            stats_mask = torch.ones_like(active)
        elif tuple(stats_mask.shape) != (batch_size, expected):
            raise ValueError(
                f"stats_mask must have shape {(batch_size, expected)}, "
                f"got {tuple(stats_mask.shape)}."
            )
        stats_mask = stats_mask.to(dtype=dtype)
        masked_stats = stats * stats_mask.unsqueeze(-1)

        if self.config.stats_only:
            numeric = torch.cat(
                (active.unsqueeze(-1), masked_stats, stats_mask.unsqueeze(-1)),
                dim=-1,
            )
        else:
            masked_parameters = parameters * parameter_mask
            numeric = torch.cat(
                (
                    active.unsqueeze(-1),
                    masked_stats,
                    masked_parameters,
                    parameter_mask,
                ),
                dim=-1,
            )

        type_ids = torch.arange(expected, device=active.device)
        type_tokens = self.type_embedding(type_ids).unsqueeze(0)
        type_tokens = type_tokens.expand(batch_size, -1, -1)

        # Convert -1 to the null order and zero-based stages to 1..K.
        order_ids = torch.where(order >= 0, order + 1, torch.zeros_like(order))
        order_ids = order_ids.clamp(0, expected).long()
        subtype_ids = subtype.clamp(0, self.config.num_subtypes - 1).long()

        hidden = type_tokens + self.numeric_projection(numeric)
        if not self.config.stats_only:
            hidden = (
                hidden
                + self.order_embedding(order_ids)
                + self.subtype_embedding(subtype_ids)
            )
        tokens = self.output_projection(hidden)
        if self.gate is not None:
            tokens = torch.tanh(self.gate) * tokens
        return tokens


def make_null_composite_condition(
    batch_size: int,
    config: CompositeConditionConfig,
    *,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """Create the unconditional counterpart used by classifier-free guidance."""
    k = len(config.degradation_types)
    p = config.parameter_dim
    return {
        "active": torch.zeros(batch_size, k, device=device),
        "stats": torch.zeros(batch_size, k, 2, device=device),
        "stats_mask": torch.zeros(batch_size, k, device=device),
        "parameters": torch.zeros(batch_size, k, p, device=device),
        "parameter_mask": torch.zeros(batch_size, k, p, device=device),
        "order": torch.full((batch_size, k), -1, device=device, dtype=torch.long),
        "subtype": torch.zeros(batch_size, k, device=device, dtype=torch.long),
    }
