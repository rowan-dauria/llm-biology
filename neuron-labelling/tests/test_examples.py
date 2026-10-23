import numpy as np
from neuron_labelling.examples import (
    pole_evidence,
    pole_stats,
    pole_strength,
    render,
    select_positions,
)


def decode(token_id: int) -> str:
    return f"t{token_id} "


def leaky_neuron():
    """Two chunks of 8 tokens; chunk 0 has a real positive firing, chunk 1 a big leak."""
    gate = np.full((2, 8), -5.0)
    value = np.zeros((2, 8))
    gate[0, 5], value[0, 5] = 4.0, 2.0  # open gate: act ~ +7.9
    gate[0, 6], value[0, 6] = 4.0, -3.0  # open gate: act ~ -11.8 (negative pole)
    gate[1, 6], value[1, 6] = -1.2, -20.0  # closed gate, negative value: act ~ +5.6 (leak)
    return gate, value


def test_pole_strength_separates_leak():
    gate, value = leaky_neuron()
    open_pos = pole_strength(gate, value, "positive", gate_open=True)
    leak_pos = pole_strength(gate, value, "positive", gate_open=False)
    assert open_pos[0, 5] > 7 and open_pos[1, 6] == 0
    assert leak_pos[1, 6] > 5 and leak_pos[0, 5] == 0
    assert pole_strength(gate, value, "negative", gate_open=True)[0, 6] > 11


def test_pole_stats_reports_leak_share():
    gate, value = leaky_neuron()
    valid = np.ones_like(gate, dtype=bool)
    stats = pole_stats(gate, value, "positive", valid, top_k=2)
    assert np.isclose(stats.max_act, 4.0 / (1 + np.exp(-4.0)) * 2.0)
    assert stats.leak_share_top == 0.5
    assert 0.6 < stats.leak_max_ratio < 0.8


def test_select_positions_one_per_chunk():
    strength = np.array([[0, 5, 9, 0], [0, 0, 3, 0], [0, 0, 0, 0]], dtype=float)
    assert select_positions(strength, 5) == [(0, 2), (1, 2)]
    rng = np.random.default_rng(0)
    assert select_positions(strength, 5, lo=1, hi=5, rng=rng) == [(1, 2)]


def test_render_marks_target_and_strong_neighbours():
    tokens = np.arange(6)
    strength = np.array([0, 0, 4, 0, 10, 0], dtype=float)
    text = render(tokens, strength, 4, 10.0, decode, left=3, right=1)
    assert text == "t1 t2 ⟨4⟩t3 [[t4 ]]⟨10⟩t5 "


def test_pole_evidence_skips_leaks_and_early_positions():
    gate, value = leaky_neuron()
    gate[1, 0], value[1, 0] = 9.0, 9.0  # huge firing at a skipped position
    tokens = np.arange(16).reshape(2, 8)
    evidence = pole_evidence(tokens, gate, value, "positive", decode, skip_positions=2)
    assert [(ex.chunk, ex.position) for ex in evidence.top] == [(0, 5)]
    assert evidence.top[0].score == 10
    assert [(ex.chunk, ex.position) for ex in evidence.leaks] == [(1, 6)]
    assert evidence.leaks[0].gate < 0
