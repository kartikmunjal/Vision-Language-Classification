import numpy as np

from vision_language_classification.calibration import fit_temperature, sigmoid
from vision_language_classification.ensemble import confidence_weighted_vote, vote_entropy
from vision_language_classification.metrics import cohens_kappa, expected_calibration_error
from vision_language_classification.rules import label_caption
from vision_language_classification.splits import assert_no_group_leakage, assign_group_splits


def test_rule_labeler_is_auditable():
    labels = label_caption("Two people are running outside with a dog at night")
    assert all(labels[name]["label"] == 1 for name in labels)


def test_kappa_known_cases():
    assert cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert np.isclose(cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)


def test_ensemble_and_entropy():
    opinions = [{"label": 1, "confidence": 0.9}, {"label": 1, "confidence": 0.7}, {"label": 0, "confidence": 0.8}]
    assert confidence_weighted_vote(opinions)["label"] == 1
    assert np.isclose(vote_entropy(opinions), 0.9182958340544896)


def test_group_split_cannot_leak():
    rows = [{"source_video_id": "a", "frame": 1}, {"source_video_id": "a", "frame": 2}, {"source_video_id": "b", "frame": 1}]
    assigned = assign_group_splits(rows)
    assert assigned[0]["split"] == assigned[1]["split"]
    assert_no_group_leakage(assigned)


def test_temperature_scaling_does_not_worsen_fit_loss():
    logits = np.array([8.0, 4.0, -5.0, -7.0])
    labels = np.array([1, 0, 0, 1])
    temperature = fit_temperature(logits, labels)
    def loss(p):
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    assert loss(sigmoid(logits / temperature)) <= loss(sigmoid(logits))


def test_ece_zero_for_balanced_half_confidence():
    assert expected_calibration_error([0, 1], [0.5, 0.5], bins=2) == 0.0
