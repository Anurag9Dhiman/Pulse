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
