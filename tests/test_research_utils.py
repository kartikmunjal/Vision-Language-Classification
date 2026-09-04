import numpy as np
from vision_language_classification.research import bootstrap_ci, paired_trial_summary, retrieval_metrics, minmax, greedy_diversity
from vision_language_classification.research.retrieval import object_disjoint_semihard


def test_paired_summary_and_bootstrap_are_deterministic():
    assert bootstrap_ci([1,2,3]) == bootstrap_ci([1,2,3])
    result=paired_trial_summary([2,3,4],[1,1,1])
    assert result["n_trials"] == 3 and result["estimate"] == 2.0


def test_retrieval_metrics_identity():
    assert retrieval_metrics(np.eye(6))["recall_at_1"] == 1.0


def test_acquisition_helpers():
    assert np.allclose(minmax([2,4]),[0,1])
    assert len(set(greedy_diversity(np.eye(4),3))) == 3


def test_object_disjoint_negative():
    sim=np.array([[1,.9,.8],[.9,1,.7],[.8,.7,1]])
    out=object_disjoint_semihard(sim,[{"person"},{"person"},{"dog"}],top_fraction=1)
    assert out[0] == 2 and out[1] == 2
