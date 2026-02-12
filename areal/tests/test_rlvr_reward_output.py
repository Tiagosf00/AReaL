import pytest
import torch

from areal.workflow.rlvr import RLVRWorkflow


def test_reward_output_number_to_tensor():
    out = RLVRWorkflow._reward_output_to_tensor(1.25)
    assert out.dtype == torch.float32
    assert out.ndim == 0
    assert float(out.item()) == pytest.approx(1.25)


def test_reward_output_scalar_tensor_to_tensor():
    out = RLVRWorkflow._reward_output_to_tensor(torch.tensor(2.5, dtype=torch.float64))
    assert out.dtype == torch.float32
    assert out.ndim == 0
    assert float(out.item()) == pytest.approx(2.5)


def test_reward_output_vector_tensor_kept():
    rewards = torch.tensor([1.0, -1.0, 3.0], dtype=torch.float64)
    out = RLVRWorkflow._reward_output_to_tensor(rewards)
    torch.testing.assert_close(out, rewards.float())


def test_reward_output_empty_tensor_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        RLVRWorkflow._reward_output_to_tensor(torch.tensor([], dtype=torch.float32))


def test_reward_output_2d_tensor_rejected():
    with pytest.raises(ValueError, match="scalar or 1D objective list"):
        RLVRWorkflow._reward_output_to_tensor(
            torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        )


def test_reward_output_list_rejected():
    with pytest.raises(TypeError):
        RLVRWorkflow._reward_output_to_tensor([1.0, 2.0])  # type: ignore[arg-type]


def test_reward_output_dict_rejected():
    with pytest.raises(TypeError):
        RLVRWorkflow._reward_output_to_tensor({"a": 1.0})  # type: ignore[arg-type]
