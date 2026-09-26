from par.safety.environment import load_profile


def test_load_simulation_profile():
    profile = load_profile("simulation")
    assert profile.name == "simulation"
    assert profile.workspace.contains(0.0, 0.0, 0.0)
    assert not profile.approval_required


def test_load_real_robot_profile():
    profile = load_profile("real_robot")
    assert profile.name == "real_robot"
    assert profile.approval_required
    assert profile.max_velocity < load_profile("simulation").max_velocity


def test_load_strict_simulation_profile():
    profile = load_profile("strict_simulation")
    assert profile.name == "strict_simulation"
    assert not profile.approval_required
    # The whole point of this profile (see evaluation/benchmarks.py): velocity,
    # not workspace, is the binding constraint.
    assert profile.max_velocity * profile.action_timeout_seconds < profile.workspace.x[1]


def test_load_high_safety_profile():
    profile = load_profile("high_safety")
    assert profile.name == "high_safety"
    assert profile.approval_required
    assert profile.max_velocity * profile.action_timeout_seconds < profile.workspace.x[1]
    assert profile.collision_margin > load_profile("simulation").collision_margin
