import torch

from composite_condition import (
    CompositeConditionConfig,
    CompositeConditionEncoder,
    GlobalStatsTokenConfig,
    GlobalStatsTokenEncoder,
    make_null_composite_condition,
)


def test_composite_condition_shape_and_gradient():
    config = CompositeConditionConfig()
    encoder = CompositeConditionEncoder(config)
    condition = make_null_composite_condition(2, config, device="cpu")

    # Activate haze and rain in application order.
    condition["active"][:, 1:3] = 1
    condition["stats"][:, 1, :] = torch.tensor([0.13, 0.04])
    condition["stats"][:, 2, :] = torch.tensor([0.09, 0.07])
    condition["parameters"][:, 1, :2] = torch.tensor([1.2, 0.75])
    condition["parameters"][:, 2, :2] = torch.tensor([0.7, -0.25])
    condition["parameter_mask"][:, 1, :2] = 1
    condition["parameter_mask"][:, 2, :2] = 1
    condition["order"][:, 1] = 0
    condition["order"][:, 2] = 1

    tokens = encoder(**condition)
    assert tokens.shape == (2, 4, 768)

    tokens.sum().backward()
    assert encoder.output_projection[-1].weight.grad is not None
    assert encoder.numeric_projection[0].weight.grad is not None
    assert encoder.type_embedding.weight.grad is not None


def test_invalid_number_of_factors_is_rejected():
    config = CompositeConditionConfig()
    encoder = CompositeConditionEncoder(config)
    condition = make_null_composite_condition(1, config, device="cpu")
    condition["active"] = condition["active"][:, :-1]

    try:
        encoder(**condition)
    except ValueError as error:
        assert "Expected 4 degradation factors" in str(error)
    else:
        raise AssertionError("Invalid factor count was not rejected")


def test_stats_only_zero_gate_starts_as_exact_vanilla_branch():
    config = CompositeConditionConfig(stats_only=True, zero_gate=True)
    encoder = CompositeConditionEncoder(config)
    condition = make_null_composite_condition(2, config, device="cpu")
    condition["active"][:, 1:3] = 1
    condition["stats"][:, 1:3] = 0.2
    condition["stats_mask"][:, 1:3] = 1

    tokens = encoder(**condition)
    assert torch.equal(tokens, torch.zeros_like(tokens))
    tokens.sum().backward()
    assert encoder.gate.grad is not None


def test_global_stats_are_two_separate_tokens_with_gradient():
    encoder = GlobalStatsTokenEncoder(GlobalStatsTokenConfig())
    stats = torch.zeros(3, 2, 129)
    stats[:, 0, 12] = 1
    stats[:, 1, 37] = 1

    tokens = encoder(stats)
    assert tokens.shape == (3, 2, 768)
    tokens.sum().backward()
    assert encoder.numeric_projection[0].weight.grad is not None
    assert encoder.stat_type_embedding.weight.grad is not None
