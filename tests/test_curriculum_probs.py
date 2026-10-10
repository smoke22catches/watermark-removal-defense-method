"""Acceptance tests for normalized attack-sampler curriculum probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.sampler import (
    BRANCH_NAMES,
    curriculum_probs,
    sample_branch_index,
    validate_curriculum_config,
)


def _default_cfg(**overrides):
    cur = {"p0": 0.1, "kappa": 0.01, "p_max": 0.5, "p_adv": 0.2}
    cur.update(overrides)
    return {"curriculum": cur}


def test_probs_nonneg_and_sum_to_one() -> None:
    cfg = _default_cfg()
    for e in (0, 10, 40, 100):
        p_dist, p_regen, p_adv = curriculum_probs(e, cfg)
        assert p_dist >= 0.0 and p_regen >= 0.0 and p_adv >= 0.0, (e, p_dist, p_regen, p_adv)
        s = p_dist + p_regen + p_adv
        assert abs(s - 1.0) < 1e-9, f"epoch {e}: sum={s}"


def test_empirical_frequencies_at_epoch_0() -> None:
    cfg = _default_cfg(p0=0.1, kappa=0.01, p_max=0.5, p_adv=0.2)
    probs = curriculum_probs(0, cfg)
    assert abs(probs[0] - 0.7) < 1e-12
    assert abs(probs[1] - 0.1) < 1e-12
    assert abs(probs[2] - 0.2) < 1e-12

    gen = torch.Generator()
    gen.manual_seed(0)
    counts = [0, 0, 0]
    n = 10_000
    for _ in range(n):
        counts[sample_branch_index(probs, gen)] += 1
    freqs = [c / n for c in counts]
    expected = (0.7, 0.1, 0.2)
    for name, got, want in zip(BRANCH_NAMES, freqs, expected):
        assert abs(got - want) < 0.02, f"{name}: emp={got}, cfg={want}"


def test_invalid_config_raises() -> None:
    try:
        validate_curriculum_config(_default_cfg(p_max=0.9, p_adv=0.2))
    except ValueError as e:
        assert "p_max" in str(e) and "p_adv" in str(e)
    else:
        raise AssertionError("expected ValueError for p_max=0.9, p_adv=0.2")

    try:
        validate_curriculum_config(_default_cfg(p0=0.6, p_max=0.5))
    except ValueError as e:
        assert "p0" in str(e)
    else:
        raise AssertionError("expected ValueError for p0 > p_max")

    try:
        validate_curriculum_config(_default_cfg(kappa=-0.01))
    except ValueError as e:
        assert "kappa" in str(e)
    else:
        raise AssertionError("expected ValueError for negative kappa")


if __name__ == "__main__":
    test_probs_nonneg_and_sum_to_one()
    test_empirical_frequencies_at_epoch_0()
    test_invalid_config_raises()
    print("=== all curriculum probability acceptance tests passed ===")
