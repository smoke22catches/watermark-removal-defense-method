"""Stochastic attack sampler over distortion / regen / PGD branches."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

BRANCH_NAMES: Tuple[str, str, str] = ("dist", "regen", "adv")

# Single source of truth for curriculum defaults.
DEFAULT_P0 = 0.1
DEFAULT_KAPPA = 0.01
DEFAULT_P_MAX = 0.5
DEFAULT_P_ADV = 0.2


def curriculum_params(cfg: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    """Return ``(p0, kappa, p_max, p_adv)`` from config (with defaults)."""
    cur = cfg.get("curriculum", {}) or {}
    return (
        float(cur.get("p0", DEFAULT_P0)),
        float(cur.get("kappa", DEFAULT_KAPPA)),
        float(cur.get("p_max", DEFAULT_P_MAX)),
        float(cur.get("p_adv", DEFAULT_P_ADV)),
    )


def validate_curriculum_config(cfg: Mapping[str, Any]) -> None:
    """Fail loudly if curriculum hyperparameters are invalid."""
    p0, kappa, p_max, p_adv = curriculum_params(cfg)
    for name, value in (("p0", p0), ("kappa", kappa), ("p_max", p_max), ("p_adv", p_adv)):
        if value < 0.0:
            raise ValueError(f"curriculum.{name} must be >= 0, got {value}")
    if p0 > p_max:
        raise ValueError(f"curriculum.p0 ({p0}) must be <= curriculum.p_max ({p_max})")
    if p_max + p_adv > 1.0 + 1e-12:
        raise ValueError(
            f"curriculum.p_max + curriculum.p_adv must be <= 1, "
            f"got {p_max} + {p_adv} = {p_max + p_adv}"
        )


def curriculum_probs(epoch: int, cfg: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Return ``(p_dist, p_regen, p_adv)`` for epoch ``e`` (0-based).

    Schedule (single source of truth)::

        p_regen(e) = min(p0 + kappa * e, p_max)
        p_dist(e)  = 1 - p_regen(e) - p_adv
        p_adv      = constant
    """
    p0, kappa, p_max, p_adv = curriculum_params(cfg)
    e = int(epoch)
    p_regen = min(p0 + kappa * e, p_max)
    p_dist = 1.0 - p_regen - p_adv
    return p_dist, p_regen, p_adv


def sample_branch_index(
    probs: Sequence[float],
    generator: torch.Generator,
) -> int:
    """Categorical draw over three normalized branch probabilities."""
    t = torch.tensor(list(probs[:3]), dtype=torch.float64)
    if t.sum().item() <= 0.0:
        raise ValueError(f"attack probabilities must be positive, got {tuple(probs[:3])}")
    return int(torch.multinomial(t, num_samples=1, generator=generator).item())


class AttackSampler(nn.Module):
    """Sample an attack from A_dist / A_regen / A_adv via a seeded categorical draw.

    ``probs`` is ``(p_dist, p_regen, p_adv)`` and must sum to 1.
    """

    def __init__(
        self,
        distortion_bank: nn.Module,
        regen_proxy: nn.Module,
        decoder: nn.Module,
        probs: Sequence[float] = (0.7, 0.1, 0.2),
        pgd_eps: float = 0.02,
        pgd_alpha: float = 0.005,
        pgd_steps: int = 5,
        seed: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        self.distortion_bank = distortion_bank
        self.regen_proxy = regen_proxy
        self.decoder = decoder
        self.probs = tuple(float(p) for p in probs[:3])
        self.pgd_eps = pgd_eps
        self.pgd_alpha = pgd_alpha
        self.pgd_steps = pgd_steps
        if generator is not None:
            self._rng = generator
        else:
            self._rng = torch.Generator()
            self._rng.manual_seed(int(seed))

    def sample_type(self) -> str:
        """Draw a branch name using the seeded categorical generator."""
        idx = sample_branch_index(self.probs, self._rng)
        return BRANCH_NAMES[idx]

    def state_dict_resume(self) -> dict:
        """Serializable sampler state for interruptible training resume."""
        return {
            "probs": tuple(float(p) for p in self.probs[:3]),
            "rng": self._rng.get_state(),
            "pgd_eps": float(self.pgd_eps),
            "pgd_alpha": float(self.pgd_alpha),
            "pgd_steps": int(self.pgd_steps),
        }

    def load_state_dict_resume(self, state: Mapping[str, Any]) -> None:
        """Restore sampler RNG and probabilities from a checkpoint."""
        if not state:
            return
        if "probs" in state and state["probs"] is not None:
            self.probs = tuple(float(p) for p in state["probs"][:3])
        if state.get("rng") is not None:
            self._rng.set_state(state["rng"])
        if "pgd_eps" in state:
            self.pgd_eps = float(state["pgd_eps"])
        if "pgd_alpha" in state:
            self.pgd_alpha = float(state["pgd_alpha"])
        if "pgd_steps" in state:
            self.pgd_steps = int(state["pgd_steps"])

    def forward(
        self,
        x_w: torch.Tensor,
        m: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, str]:
        attack_type = self.sample_type()
        if attack_type == "dist":
            return self.distortion_bank(x_w), "dist"
        if attack_type == "regen":
            return self.regen_proxy(x_w, text_embeds), "regen"
        # Marker only: train_step runs PGD once, then STE-attaches to x_w.
        _ = (m, self.decoder, self.pgd_eps, self.pgd_alpha, self.pgd_steps)
        return x_w, "adv"
