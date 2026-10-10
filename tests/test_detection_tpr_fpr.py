"""Acceptance tests for analytic Binomial TPR@FPR detection metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.metrics import (
    analytic_binom_threshold,
    binom_tail_prob,
    compute_detection_report,
    matched_bit_counts,
)


def test_analytic_threshold_k100_fpr_1e3() -> None:
    k, fpr = 100, 1e-3
    t = analytic_binom_threshold(k, fpr)
    assert binom_tail_prob(k, t) <= fpr + 1e-15
    if t > 0:
        assert binom_tail_prob(k, t - 1) > fpr
    print(f"[ok] analytic threshold k={k} FPR={fpr}: t={t} "
          f"P(X>=t)={binom_tail_prob(k, t):.6g} "
          f"P(X>=t-1)={binom_tail_prob(k, t - 1):.6g}")


class RandomBitDecoder(nn.Module):
    """Decoder that ignores the image and emits uniform random logits."""

    def __init__(self, msg_len: int) -> None:
        super().__init__()
        self.msg_len = msg_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        # Large-magnitude random logits → hard bits ~ Bern(0.5)
        bits = torch.randint(0, 2, (b, self.msg_len), device=x.device).float()
        return (bits * 20.0) - 10.0


class PerfectDecoder(nn.Module):
    """Stores the last embedded message and returns it as logits."""

    def __init__(self) -> None:
        super().__init__()
        self._msg: torch.Tensor | None = None

    def set_message(self, m: torch.Tensor) -> None:
        self._msg = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._msg is not None
        m = self._msg.to(x.device)
        if m.size(0) != x.size(0):
            m = m.expand(x.size(0), -1)
        return (m * 20.0) - 10.0


def test_empirical_fpr_random_decoder() -> None:
    torch.manual_seed(0)
    k = 100
    fpr = 1e-3
    n = 20_000
    dec = RandomBitDecoder(k)
    x = torch.zeros(n, 3, 8, 8)
    m = torch.randint(0, 2, (n, k)).float()
    with torch.no_grad():
        logits = dec(x)
        stats = matched_bit_counts(logits, m).tolist()
    t = analytic_binom_threshold(k, fpr)
    emp = sum(1 for s in stats if s >= t) / n
    lo, hi = fpr / 2.0, 2.0 * fpr
    assert lo <= emp <= hi, (
        f"empirical FPR {emp:.6g} not within factor 2 of nominal {fpr} ([{lo}, {hi}])"
    )
    print(f"[ok] random decoder n={n}: t={t} emp_FPR={emp:.6g} nominal={fpr}")


def test_perfect_decoding_tpr_is_one() -> None:
    torch.manual_seed(1)
    k = 64
    n = 128
    fpr = 1e-3
    dec = PerfectDecoder()
    m = torch.randint(0, 2, (n, k)).float()
    dec.set_message(m)
    x_w = torch.randn(n, 3, 8, 8)
    x_clean = torch.randn(n, 3, 8, 8)
    with torch.no_grad():
        pos = matched_bit_counts(dec(x_w), m).tolist()
        # Clean: random reference messages → null Bin(k, 0.5) under perfect decoder
        # that still returns the *watermark* message would be biased; use random bits.
        m_ref = torch.randint(0, 2, (n, k)).float()
        # For clean empirical null under analytic assumption, use random decoder output.
        rand = RandomBitDecoder(k)
        clean = matched_bit_counts(rand(x_clean), m_ref).tolist()
    assert all(s == k for s in pos)
    report = compute_detection_report(pos, clean, msg_len=k, target_fpr=fpr)
    assert report.tpr_analytic == 1.0
    assert report.tpr_empirical == 1.0
    print(f"[ok] perfect decoding TPR={report.tpr_analytic} at t={report.analytic_threshold}")


if __name__ == "__main__":
    test_analytic_threshold_k100_fpr_1e3()
    test_empirical_fpr_random_decoder()
    test_perfect_decoding_tpr_is_one()
    print("=== all detection TPR@FPR acceptance tests passed ===")
