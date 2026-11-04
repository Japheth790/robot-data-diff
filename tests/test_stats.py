import numpy as np

from robot_data_diff.stats import action_quality


def test_constant_actions_are_all_idle():
    idle, spikes = action_quality(np.ones((50, 3)))
    assert idle == 1.0
    assert spikes == 0


def test_single_injected_jump_is_one_spike():
    a = np.stack([np.linspace(0, 1, 100), np.linspace(0, -1, 100)], axis=1)
    a[60:] += 5.0
    idle, spikes = action_quality(a)
    assert spikes == 1
    assert idle == 0.0


def test_smooth_motion_has_no_spikes():
    a = np.linspace(0, 1, 100)[:, None]
    assert action_quality(a) == (0.0, 0)


def test_single_frame_episode():
    assert action_quality(np.zeros((1, 4))) == (0.0, 0)


def test_one_dimensional_input_is_accepted():
    assert action_quality(np.zeros(10))[0] == 1.0
