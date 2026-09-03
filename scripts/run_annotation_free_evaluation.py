#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from vision_language_classification.analysis import blur_band, caption_length_band
from vision_language_classification.calibration import fit_temperature, reliability_bins, sigmoid
from vision_language_classification.metrics import (
    accuracy,
    bootstrap_ci,
    cohens_kappa,
    expected_calibration_error,
    metric_with_ci,
)
from vision_language_classification.schema import read_jsonl

TASKS = ("multiple_subjects", "human_present", "animal_present")


def indexed(path):
    return {row["example_id"]: row for row in read_jsonl(path)}


def probability(row, task):
    item = row["labels"][task]
    return float(item.get("probability", item["label"]))


def probability_metric_with_ci(y, p, metric, replicates, seed):
    return metric_with_ci(y, p, metric, replicates=replicates, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("oracle")
    parser.add_argument("rules")
    parser.add_argument("llm")
    parser.add_argument("clip")
    parser.add_argument("ensemble")
    parser.add_argument("logits")
    parser.add_argument("output")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = indexed(args.manifest)
    oracle = indexed(args.oracle)
    sources = {name: indexed(path) for name, path in (("rules", args.rules), ("llm", args.llm), ("clip", args.clip), ("ensemble", args.ensemble))}
    logits = indexed(args.logits)
    report = {
        "ground_truth": "COCO 2017 human instance annotations (silver)",
        "verifiable_tasks": list(TASKS),
        "excluded_from_ground_truth_claims": ["outdoor", "dynamic_scene", "night"],
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_repetitions": args.random_repetitions,
        "n_trials": {"models": 1, "active_correction_budgets": 3},
        "tasks": {},
    }
    test_ids = sorted(item for item in oracle if manifest[item]["split"] == "test")
    calibration_ids = sorted(item for item in oracle if manifest[item]["split"] == "calibration")
    for task in TASKS:
        task_report = {"weak_sources": {}, "classifier": {}, "slices": {}}
        y_test = np.array([oracle[item]["labels"][task] for item in test_ids])
        for source_name, rows in sources.items():
            predictions = np.array([rows[item]["labels"][task]["label"] for item in test_ids])
            task_report["weak_sources"][source_name] = {
                "accuracy": metric_with_ci(y_test, predictions, accuracy, replicates=args.bootstrap_replicates, seed=args.seed),
                "cohens_kappa": metric_with_ci(y_test, predictions, cohens_kappa, replicates=args.bootstrap_replicates, seed=args.seed),
            }
        cal_logits = np.array([logits[item]["logits"][task] for item in calibration_ids])
        cal_y = np.array([oracle[item]["labels"][task] for item in calibration_ids])
        test_logits = np.array([logits[item]["logits"][task] for item in test_ids])
        temperature = fit_temperature(cal_logits, cal_y)
        before, after = sigmoid(test_logits), sigmoid(test_logits / temperature)
        for name, probs in (("before_temperature", before), ("after_temperature", after)):
            task_report["classifier"][name] = {
                "accuracy": probability_metric_with_ci(y_test, probs, lambda y, p: accuracy(y, p >= 0.5), args.bootstrap_replicates, args.seed),
                "ece": probability_metric_with_ci(y_test, probs, expected_calibration_error, args.bootstrap_replicates, args.seed),
                "reliability_bins": reliability_bins(y_test, probs),
            }
        task_report["classifier"]["temperature"] = temperature
        slice_values = {
            "blur_band": [blur_band(manifest[item].get("blur_score")) for item in test_ids],
            "caption_length": [caption_length_band(manifest[item]["caption"]) for item in test_ids],
            "source_category": [manifest[item]["source_category"] for item in test_ids],
        }
        for dimension, values in slice_values.items():
            task_report["slices"][dimension] = {}
            for value in sorted(set(values)):
                mask = np.array([item == value for item in values])
                if mask.sum() < 20:
                    continue
                task_report["slices"][dimension][value] = {
                    "n": int(mask.sum()),
                    "accuracy": metric_with_ci(y_test[mask], after[mask] >= 0.5, accuracy, replicates=args.bootstrap_replicates, seed=args.seed),
                    "ece": probability_metric_with_ci(y_test[mask], after[mask], expected_calibration_error, args.bootstrap_replicates, args.seed),
                }
        report["tasks"][task] = task_report

    train_ids = sorted(item for item in oracle if manifest[item]["split"] == "train")
    entropy = np.array([sources["ensemble"][item]["vote_entropy"] for item in train_ids])
    rng = np.random.default_rng(args.seed)
    report["active_correction"] = {}
    for budget in (0.05, 0.10, 0.20):
        count = max(1, round(len(train_ids) * budget))
        targeted = np.argsort(-entropy, kind="stable")[:count]
        budget_report = {"corrected_examples": count, "tasks": {}}
        for task in TASKS:
            y = np.array([oracle[item]["labels"][task] for item in train_ids])
            base = np.array([probability(sources["ensemble"][item], task) for item in train_ids])
            targeted_p = base.copy(); targeted_p[targeted] = y[targeted]
            targeted_gain = {
                "accuracy": accuracy(y, targeted_p >= 0.5) - accuracy(y, base >= 0.5),
                "ece": expected_calibration_error(y, base) - expected_calibration_error(y, targeted_p),
            }
            random_gains = {"accuracy": [], "ece": []}
            for _ in range(args.random_repetitions):
                chosen = rng.choice(len(train_ids), count, replace=False)
                random_p = base.copy(); random_p[chosen] = y[chosen]
                random_gains["accuracy"].append(accuracy(y, random_p >= 0.5) - accuracy(y, base >= 0.5))
                random_gains["ece"].append(expected_calibration_error(y, base) - expected_calibration_error(y, random_p))
            comparison = {}
            for metric in ("accuracy", "ece"):
                differences = targeted_gain[metric] - np.array(random_gains[metric])
                low, high = bootstrap_ci(differences, replicates=args.bootstrap_replicates, seed=args.seed)
                comparison[metric] = {"targeted_gain": targeted_gain[metric], "random_mean_gain": float(np.mean(random_gains[metric])), "targeted_minus_random": float(np.mean(differences)), "ci95": [low, high]}
            budget_report["tasks"][task] = comparison
        report["active_correction"][str(budget)] = budget_report
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
