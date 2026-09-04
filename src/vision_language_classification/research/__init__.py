"""Reusable research utilities for cross-modal data selection and evaluation."""

from .statistics import bootstrap_ci, paired_trial_summary
from .retrieval import retrieval_metrics
from .acquisition import minmax, greedy_diversity

__all__ = ["bootstrap_ci", "paired_trial_summary", "retrieval_metrics", "minmax", "greedy_diversity"]
