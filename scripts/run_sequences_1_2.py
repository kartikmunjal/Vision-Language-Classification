#!/usr/bin/env python3
"""Run the cross-modal evidence benchmark and preference reward experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [11, 22, 33, 44, 55]
FEATURES = ["clip_score", "lpips_temporal", "motion_smoothness", "fvd_score"]


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bootstrap(values: np.ndarray, seed: int = 20260903, n: int = 10_000):
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), (n, len(values)))], axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def preference_rows(pref):
    rows = []
    for pair in pref["pairs"]:
        choice = pair["majority_choice"]
        if choice not in {"a", "b"}:
            continue
        a = pair["automated_scores"]["video_a"]
        b = pair["automated_scores"]["video_b"]
        rows.append(
            {
                "id": pair["id"],
                "group": " ".join(pair["prompt"].lower().split()),
                "x": [float(a[k]) - float(b[k]) for k in FEATURES],
                "y": int(choice == "a"),
                "composite_delta": float(a["composite"]) - float(b["composite"]),
            }
        )
    return rows


def reward_experiment(pref):
    rows = preference_rows(pref)
    x = np.asarray([r["x"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    groups = np.asarray([r["group"] for r in rows])
    comp_prob = 1 / (1 + np.exp(-np.asarray([r["composite_delta"] for r in rows]) * 10))
    comp_pred = (comp_prob >= 0.5).astype(int)
    learned_prob = np.zeros(len(rows))
    fold_id = np.full(len(rows), -1)
    splitter = GroupKFold(n_splits=5)
    fold_models = []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEEDS[fold]))
        model.fit(x[train], y[train])
        learned_prob[test] = model.predict_proba(x[test])[:, 1]
        fold_id[test] = fold
        fold_models.append({"fold": fold, "seed": SEEDS[fold], "n_train": len(train), "n_test": len(test)})
    learned_pred = (learned_prob >= 0.5).astype(int)
    learned_correct = (learned_pred == y).astype(float)
    comp_correct = (comp_pred == y).astype(float)
    delta = learned_correct - comp_correct
    rng = np.random.default_rng(20260903)
    idx = rng.integers(0, len(y), (10_000, len(y)))
    auc_delta = []
    for sample in idx:
        if len(np.unique(y[sample])) == 2:
            auc_delta.append(roc_auc_score(y[sample], learned_prob[sample]) - roc_auc_score(y[sample], comp_prob[sample]))
    per_pair = [
        {"id": r["id"], "fold": int(fold_id[i]), "target": int(y[i]), "learned_probability_a": float(learned_prob[i]),
         "handcrafted_probability_a": float(comp_prob[i])}
        for i, r in enumerate(rows)
    ]
    return {
        "n_pairs": len(rows), "n_trials": 5, "folds": fold_models, "features": FEATURES,
        "learned": {"accuracy": float(accuracy_score(y, learned_pred)), "roc_auc": float(roc_auc_score(y, learned_prob)),
                    "brier": float(brier_score_loss(y, learned_prob))},
        "handcrafted": {"accuracy": float(accuracy_score(y, comp_pred)), "roc_auc": float(roc_auc_score(y, comp_prob)),
                        "brier": float(brier_score_loss(y, comp_prob))},
        "paired_deltas_learned_minus_handcrafted": {
            "accuracy": {"estimate": float(delta.mean()), "ci95": bootstrap(delta)},
            "roc_auc": {"estimate": float(roc_auc_score(y, learned_prob) - roc_auc_score(y, comp_prob)),
                        "ci95": [float(x) for x in np.quantile(auc_delta, [0.025, 0.975])]},
            "brier": {"estimate": float(brier_score_loss(y, learned_prob) - brier_score_loss(y, comp_prob))},
        },
        "per_pair_predictions": per_pair,
    }


def evidence_rows(audio, crawl, vl, stage3, pref):
    cur = audio["contrasts"]["curation"]["metrics"]["overall"]
    aug = crawl["comparisons"]["openslr31"]["overall"]["augmented_minus_control"]
    rows = [
        {"modality": "audio", "study": "curation", "metric": "WER delta", "estimate": cur["mean_paired_delta_treatment_minus_control"],
         "ci95": cur["paired_delta_trial_bootstrap_95_ci"], "n_trials": 5, "direction": "lower_is_better"},
        {"modality": "audio", "study": "crawler augmentation", "metric": "WER delta", "estimate": aug["estimate"],
         "ci95": [aug["ci_low"], aug["ci_high"]], "n_trials": aug["n_trials"], "direction": "lower_is_better"},
        {"modality": "vision-language", "study": "powered low-quality slice", "metric": "disagreement delta", "estimate": stage3["result"]["estimate"],
         "ci95": stage3["result"]["ci95"], "n_trials": stage3["result"]["n_trials"], "direction": "lower_is_better"},
    ]
    for task, d in vl["tasks"].items():
        metric = d["weak_sources"]["ensemble"]["accuracy"]
        rows.append({"modality": "vision-language", "study": task, "metric": "ensemble accuracy", "estimate": metric["estimate"],
                     "ci95": metric["ci95"], "n_trials": metric["n_trials"], "direction": "higher_is_better"})
    rows.append({"modality": "video", "study": "human preference agreement", "metric": "Cohen kappa", "estimate": pref["metadata"]["cohen_kappa"],
                 "ci95": None, "n_trials": 1, "direction": "higher_is_better", "note": "source artifact does not provide CI"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abc-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/research_sequence"))
    args = ap.parse_args()
    sources = {
        "audio_downstream": args.abc_root / "Audio-Data-Creation/experiments/results/downstream_study/summary.json",
        "audio_crawl": args.abc_root / "Audio-Data-Creation/experiments/results/crawl_training_study/summary.json",
        "preferences": args.abc_root / "Video-Quality-Reward-Modeling/data/human_preferences/preferences.json",
        "vl_eval": args.abc_root / "Vision-Language-Classification/results/annotation_free_evaluation.json",
        "vl_stage3": args.abc_root / "Vision-Language-Classification/results/stage3_powered_followup.json",
    }
    data = {k: load(v) for k, v in sources.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {k: {"path": str(v), "sha256": sha256(v)} for k, v in sources.items()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    common = {"schema_version": 1, "plan": "RESEARCH_SEQUENCE_PLAN.md", "git_commit": commit,
              "python": platform.python_version(), "sources": provenance}
    benchmark = dict(common, sequence=1, evidence=evidence_rows(data["audio_downstream"], data["audio_crawl"], data["vl_eval"], data["vl_stage3"], data["preferences"]))
    reward = dict(common, sequence=2, result=reward_experiment(data["preferences"]))
    (args.output_dir / "sequence1_unified_benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (args.output_dir / "sequence2_reward_model.json").write_text(json.dumps(reward, indent=2) + "\n")


if __name__ == "__main__":
    main()
