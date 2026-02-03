"""Unit tests for workflow detection logic in PPOTrainer."""

import pytest

from areal.api.workflow_api import AgentWorkflow, RolloutWorkflow


class DummyRolloutWorkflow(RolloutWorkflow):
    """Test RolloutWorkflow implementation."""

    async def arun_episode(self, engine, data):
        return {}


class DummyAgentWorkflow(AgentWorkflow):
    """Test AgentWorkflow implementation (deprecated)."""

    async def run(self, data, **extra_kwargs):
        return 1.0


class DummyAgentNonInheriting:
    """No inheritance - duck typing only."""

    async def run(self, data, **extra_kwargs):
        return 1.0


class TestWorkflowDetection:
    """Test the _requires_proxy_workflow detection logic."""

    @pytest.fixture
    def trainer_with_detection(self):
        """Create a minimal object with the detection method."""
        # We can't easily instantiate a full PPOTrainer in unit tests,
        # so we'll create a simple object with just the method we need
        from areal.experimental.trainer.rl import PPOTrainer

        class MinimalTrainer:
            def _requires_proxy_workflow(self, workflow):
                # Copy the implementation from PPOTrainer
                return PPOTrainer._requires_proxy_workflow(self, workflow)

        return MinimalTrainer()

    def test_rollout_workflow_instance_no_proxy(self, trainer_with_detection):
        """RolloutWorkflow instances should NOT require proxy."""
        workflow = DummyRolloutWorkflow()
        assert not trainer_with_detection._requires_proxy_workflow(workflow)

    def test_rollout_workflow_class_no_proxy(self, trainer_with_detection):
        """RolloutWorkflow classes should NOT require proxy."""
        assert not trainer_with_detection._requires_proxy_workflow(DummyRolloutWorkflow)

    def test_rollout_workflow_string_no_proxy(self, trainer_with_detection):
        """String paths to RolloutWorkflow should NOT require proxy."""
        workflow = "areal.workflow.rlvr.RLVRWorkflow"
        assert not trainer_with_detection._requires_proxy_workflow(workflow)

    def test_agent_workflow_instance_requires_proxy(self, trainer_with_detection):
        """AgentWorkflow instances should require proxy."""
        workflow = DummyAgentWorkflow()
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_agent_workflow_class_requires_proxy(self, trainer_with_detection):
        """AgentWorkflow classes should require proxy."""
        assert trainer_with_detection._requires_proxy_workflow(DummyAgentWorkflow)

    def test_non_inheriting_agent_instance_requires_proxy(self, trainer_with_detection):
        """Non-inheriting agent instances should require proxy."""
        workflow = DummyAgentNonInheriting()
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_non_inheriting_agent_class_requires_proxy(self, trainer_with_detection):
        """Non-inheriting agent classes should require proxy."""
        assert trainer_with_detection._requires_proxy_workflow(DummyAgentNonInheriting)

    def test_invalid_string_requires_proxy(self, trainer_with_detection):
        """Invalid string paths should require proxy (fail-safe)."""
        workflow = "nonexistent.module.Workflow"
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_string_to_agent_workflow_requires_proxy(self, trainer_with_detection):
        """String paths to agent workflows should require proxy."""
        workflow = "areal.tests.experimental.openai.utils.SimpleAgent"
        assert trainer_with_detection._requires_proxy_workflow(workflow)

    def test_string_to_non_inheriting_agent_requires_proxy(
        self, trainer_with_detection
    ):
        """String paths to non-inheriting agents should require proxy."""
        workflow = "areal.tests.experimental.openai.utils.SimpleAgentNonInheriting"
        assert trainer_with_detection._requires_proxy_workflow(workflow)
