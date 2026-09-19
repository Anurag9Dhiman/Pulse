from par.core.capability import Capability, RiskLevel


def test_capability_defaults():
    cap = Capability(name="move", description="move the robot")
    assert cap.risk == RiskLevel.LOW
    assert cap.env_profiles == ["simulation"]
    assert cap.supports_rollback is False
    assert cap.execution_timeout_seconds is None
