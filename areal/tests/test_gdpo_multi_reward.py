from unittest.mock import MagicMock

import torch

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import concat_padded_tensors


def _build_actor(
    *,
    multi_reward_norm: NormConfig | None = None,
    gdpo_weights: list[float] | None = None,
) -> PPOActor:
    config = PPOActorConfig(
        multi_reward_norm=multi_reward_norm,
        gdpo_weights=gdpo_weights or [],
        kl_ctl=0.0,
        reward_scaling=1.0,
        reward_bias=0.0,
        reward_clip=20.0,
    )
    return PPOActor(config=config, engine=MagicMock())


def test_multi_reward_norm_with_equal_weights():
    rewards = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [2.0, 20.0],
            [6.0, 28.0],
        ],
        dtype=torch.float32,
    )
    actor = _build_actor(
        multi_reward_norm=NormConfig(
            mean_level="group",
            std_level="group",
            group_size=2,
        ),
    )
    output = actor._collapse_reward_objectives(rewards)

    expected = torch.tensor(
        [-1.4142135, 1.4142135, -1.4142135, 1.4142135], dtype=torch.float32
    )
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_multi_reward_norm_with_explicit_weights():
    rewards = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [2.0, 20.0],
            [6.0, 28.0],
        ],
        dtype=torch.float32,
    )
    actor = _build_actor(
        multi_reward_norm=NormConfig(
            mean_level="group",
            std_level="group",
            group_size=2,
        ),
        gdpo_weights=[1.0, 2.0],
    )
    output = actor._collapse_reward_objectives(rewards)

    expected = torch.tensor(
        [-2.1213202, 2.1213202, -2.1213202, 2.1213202], dtype=torch.float32
    )
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_multi_reward_without_multi_reward_norm_skips_objective_normalization():
    rewards = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [2.0, 20.0],
            [6.0, 28.0],
        ],
        dtype=torch.float32,
    )
    actor = _build_actor(multi_reward_norm=None, gdpo_weights=[])
    output = actor._collapse_reward_objectives(rewards)
    expected = torch.tensor([11.0, 17.0, 22.0, 34.0], dtype=torch.float32)
    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)


def test_scalar_reward_path_unchanged():
    rewards = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    actor = _build_actor(multi_reward_norm=None, gdpo_weights=[])
    output = actor._collapse_reward_objectives(rewards)
    torch.testing.assert_close(output, rewards, atol=1e-6, rtol=1e-6)


def test_weight_length_mismatch_raises():
    actor = _build_actor(multi_reward_norm=None, gdpo_weights=[1.0, 2.0, 3.0])
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    try:
        actor._collapse_reward_objectives(rewards)
        assert False, "Expected ValueError due to mismatched gdpo_weights length."
    except ValueError as e:
        assert "gdpo_weights has length" in str(e)


def test_multi_reward_norm_group_size_must_divide_batch():
    actor = _build_actor(
        multi_reward_norm=NormConfig(
            mean_level="group",
            std_level="group",
            group_size=3,
        ),
        gdpo_weights=[],
    )
    rewards = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=torch.float32,
    )

    try:
        actor._collapse_reward_objectives(rewards)
        assert False, "Expected ValueError due to invalid group_size for batch."
    except ValueError as e:
        assert "not divisible by multi_reward_norm.group_size" in str(e)


def test_compute_advantages_collapses_multi_objective_rewards():
    actor = _build_actor(
        multi_reward_norm=NormConfig(
            mean_level="group",
            std_level="group",
            group_size=2,
        ),
        gdpo_weights=[1.0, 1.0],
    )

    data = {
        "input_ids": torch.zeros((4, 3), dtype=torch.int32),
        "attention_mask": torch.ones((4, 3), dtype=torch.bool),
        "loss_mask": torch.tensor(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.int32,
        ),
        "logprobs": torch.zeros((4, 3), dtype=torch.float32),
        "rewards": torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 14.0],
                [2.0, 20.0],
                [6.0, 28.0],
            ],
            dtype=torch.float32,
        ),
    }

    out = actor.compute_advantages(data)
    expected = torch.tensor(
        [-1.4142135, 1.4142135, -1.4142135, 1.4142135], dtype=torch.float32
    )
    torch.testing.assert_close(out["rewards"], expected, atol=1e-5, rtol=1e-5)


def test_concat_padded_tensors_keeps_multi_reward_width():
    d1 = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.int32),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool
        ),
        "loss_mask": torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((2, 4), dtype=torch.float32),
        "rewards": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
    }
    d2 = {
        "input_ids": torch.tensor([[8, 9, 10]], dtype=torch.int32),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.int32),
        "logprobs": torch.zeros((1, 3), dtype=torch.float32),
        "rewards": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
    }

    out = concat_padded_tensors([d1, d2])
    assert out["rewards"].shape == (3, 2)
    torch.testing.assert_close(
        out["rewards"],
        torch.tensor(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            dtype=torch.float32,
        ),
    )
