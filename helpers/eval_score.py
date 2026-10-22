"""Live sweep score for W&B (``--eval_score_ref``, LESRL optuna_sweep).

    s_k = (P_k - G_k) / (L_k - G_k)     P_k = eval/wd/<k>/mean_power
    eval/score/mean = mean_k s_k        (1.0 = LUT level, 0 = greedy)

G_k / L_k are the measured greedy / lut3 powers of the sweep's baselines.json
(``per_spec`` layout). Same formula as optuna_sweep/objective.score_record but
UNCLAMPED: the worker clamps its objective at -1, the curve shows the raw value.
Pure python on purpose (no torch), so the LESRL tests can import it standalone.
"""
from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

ScoreRef = Dict[str, Tuple[float, float]]


def load_score_ref(path: str) -> ScoreRef:
    """{spec: (greedy, lut3)} in the baselines' spec order."""
    with open(path) as f:
        b = json.load(f)
    specs = list(b.get("specs") or b["per_spec"].keys())
    return {k: (float(b["per_spec"][k]["greedy"]), float(b["per_spec"][k]["lut3"])) for k in specs}


def eval_scores(eval_dict: dict, ref: Optional[ScoreRef]) -> Dict[str, float]:
    """eval/score/<spec> for every spec present in ``eval_dict``; eval/score/mean only
    when all of them are (a partial mean would not be the sweep's score)."""
    if not ref:
        return {}
    out: Dict[str, float] = {}
    for k, (g, l) in ref.items():
        p = eval_dict.get(f"eval/wd/{k}/mean_power")
        if p is not None:
            out[f"eval/score/{k}"] = (float(p) - g) / (l - g)
    if len(out) == len(ref):
        out["eval/score/mean"] = sum(out.values()) / len(out)
    return out
