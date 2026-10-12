"""Acceptance tests for seen regeneration-class eval entries (regen_proxy / guided_regen)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks.regeneration import build_regen_proxy
from src.eval.generative import DEFAULT_EVAL_DIFFUSION_ID, DEFAULT_TRAIN_DIFFUSION_ID
from src.eval.registry import (
    ATTACK_REGISTRY,
    apply_spec,
    get_attack,
    protocol_table_rows,
    regeneration_seen_unseen,
)
from src.utils.viz import generate_g1


class TinyDec(torch.nn.Module):
    def __init__(self, msg_len: int = 16) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, msg_len),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def _cfg() -> dict:
    return {
        "regen": {
            "use_placeholder": True,
            "sd_model_id": DEFAULT_TRAIN_DIFFUSION_ID,
            "sd_revision": None,
            "n_steps": 4,
            "t_start": 0.3,
            "eps_guided": 0.02,
            "beta_guided": 0.005,
            "J_guided": 2,
        },
        "eval": {
            "diffpure_model_id": DEFAULT_EVAL_DIFFUSION_ID,
            "regen_proxy_t_starts": [0.2, 0.3, 0.4],
            "guided_eps_values": [0.01, 0.02, 0.04],
        },
        "pgd": {"alpha": 0.005, "steps": 2},
    }


def test_regen_proxy_and_guided_regen_batch_shape() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    m = torch.randint(0, 2, (2, 16)).float()
    decoder = TinyDec(16)
    cfg = _cfg()
    proxy = build_regen_proxy(
        use_placeholder=True,
        sd_model_id=DEFAULT_TRAIN_DIFFUSION_ID,
        n_steps=4,
        t_start=0.3,
        device="cpu",
    )
    for aid in ("regen_proxy", "guided_regen"):
        spec = get_attack(aid)
        y = apply_spec(
            spec,
            x,
            strength=float(spec.strengths[len(spec.strengths) // 2]),
            message=m,
            decoder=decoder,
            regen_proxy=proxy,
            config=cfg,
        )
        assert y.shape == x.shape, f"{aid}: shape {y.shape} != {x.shape}"
        assert float(y.min()) >= -1.0 - 1e-4 and float(y.max()) <= 1.0 + 1e-4
        print(f"[ok] {aid} batch shape / range")


def test_regen_proxy_model_matches_train_diffpure_differs() -> None:
    cfg = _cfg()
    proxy = build_regen_proxy(
        use_placeholder=True,
        sd_model_id=DEFAULT_TRAIN_DIFFUSION_ID,
        device="cpu",
    )
    assert str(proxy.sd_model_id) == str(cfg["regen"]["sd_model_id"])
    assert str(proxy.sd_model_id) == DEFAULT_TRAIN_DIFFUSION_ID
    assert DEFAULT_EVAL_DIFFUSION_ID != DEFAULT_TRAIN_DIFFUSION_ID
    # apply_spec asserts train id equality and that diffpure id differs
    x = torch.rand(2, 3, 16, 16) * 2 - 1
    y = apply_spec(
        get_attack("regen_proxy"),
        x,
        strength=0.3,
        regen_proxy=proxy,
        config=cfg,
    )
    assert y.shape == x.shape
    # Wrong proxy id must abort
    bad = build_regen_proxy(
        use_placeholder=True,
        sd_model_id=DEFAULT_EVAL_DIFFUSION_ID,
        device="cpu",
    )
    try:
        apply_spec(get_attack("regen_proxy"), x, regen_proxy=bad, config=cfg)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when proxy id ≠ training id"
    print("[ok] regen_proxy uses training diffusion id; diffpure id differs")


def test_protocol_table_regen_seen_unseen_and_knowledge() -> None:
    rows = {r["id"]: r for r in protocol_table_rows()}
    assert "regen_proxy" in rows and "guided_regen" in rows
    assert rows["regen_proxy"]["seen_in_training"] is True
    assert rows["guided_regen"]["seen_in_training"] is True
    assert rows["regen_proxy"]["knowledge_level"] == "black"
    assert rows["guided_regen"]["knowledge_level"] == "white"
    assert rows["regen_proxy"]["label_uk"] == "Проксі регенерації"
    assert rows["guided_regen"]["label_uk"] == "Керована регенерація"

    for aid in ("diffpure", "rinse_2", "rinse_4", "bmshj2018", "cheng2020"):
        assert rows[aid]["class"] == "regen"
        assert rows[aid]["seen_in_training"] is False
        assert rows[aid]["knowledge_level"] == "black", aid

    # Only regen_proxy + guided_regen are seen within regen class
    regen_rows = [r for r in rows.values() if r["class"] == "regen"]
    seen = [r["id"] for r in regen_rows if r["seen_in_training"]]
    unseen = [r["id"] for r in regen_rows if not r["seen_in_training"] and not r["not_evaluated"]]
    assert set(seen) == {"regen_proxy", "guided_regen"}
    assert len(unseen) >= 1
    print(f"[ok] protocol table regen seen={seen} unseen={unseen}")


def test_g1_regen_groups_nonempty() -> None:
    groups = regeneration_seen_unseen()
    assert groups["seen"], "regen seen group must be non-empty"
    assert groups["unseen"], "regen unseen group must be non-empty"
    assert "regen_proxy" in groups["seen"] and "guided_regen" in groups["seen"]
    assert "diffpure" in groups["unseen"]

    with tempfile.TemporaryDirectory() as tmp:
        arts = generate_g1([], Path(tmp))
        assert arts.get("regen_seen")
        assert arts.get("regen_unseen")
        assert set(arts["regen_seen"]) == set(groups["seen"])
        assert set(arts["regen_unseen"]) == set(groups["unseen"])
    print(
        f"[ok] G1 regen seen={arts['regen_seen']} unseen={arts['regen_unseen']}"
    )


def test_regen_alias() -> None:
    assert get_attack("regen").id == "regen_proxy"


if __name__ == "__main__":
    test_regen_proxy_and_guided_regen_batch_shape()
    test_regen_proxy_model_matches_train_diffpure_differs()
    test_protocol_table_regen_seen_unseen_and_knowledge()
    test_g1_regen_groups_nonempty()
    test_regen_alias()
    print("=== all seen-regen eval acceptance tests passed ===")
