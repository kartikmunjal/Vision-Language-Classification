from vision_language_classification.analysis import (
    blur_band,
    correction_indices,
    disagreement_difference,
    training_texture_cut,
)


def test_locked_blur_bands():
    assert blur_band(39.99) == "low"
    assert blur_band(40) == "medium"
    assert blur_band(80) == "high"


def test_texture_cut_train_only():
    rows = [
        {"split": "train", "texture_score": value} for value in [1, 2, 3, 4]
    ] + [{"split": "test", "texture_score": -100}]
    assert training_texture_cut(rows) == 1.75


def test_targeted_correction_takes_highest_entropy():
    assert list(correction_indices([0.1, 0.9, 0.5, 0.2], 0.5, strategy="targeted", seed=1)) == [1, 2]


def test_disagreement_difference_direction():
    rows = [
        {"blur_score": 10, "texture_score": 1, "vote_entropy": 1.0},
        {"blur_score": 20, "texture_score": 1, "vote_entropy": 0.9},
        {"blur_score": 100, "texture_score": 9, "vote_entropy": 0.1},
        {"blur_score": 100, "texture_score": 9, "vote_entropy": 0.2},
    ]
    result = disagreement_difference(rows, texture_cut=2, replicates=100, seed=4)
    assert result["estimate"] > 0
    assert result["n_trials"] == 1
