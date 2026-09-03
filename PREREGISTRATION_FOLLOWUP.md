# Stage 3 Larger-Sample Follow-up Preregistration

This follow-up preserves the original Stage 3 result as the primary historical
result. It does not alter the locked quality thresholds after observing the
outcome.

## Design

- Source: COCO train2017, disjoint from the original val2017 study.
- Ordering: deterministic shuffle with seed 2026.
- Low sharpness: Laplacian variance below 40, unchanged.
- Low texture: at or below the original train-fitted cutoff recorded in
  `results/stage3_preregistered_hypothesis.json`, unchanged.
- Sampling stop: the first 64 qualifying images in shuffled order, retaining
  every comparison image scanned before the stopping point.
- Labelers: the locked rule patterns, Qwen2.5-1.5B-Instruct greedy extractor,
  and OpenCLIP ViT-B-32/laion2b_s34b_b79k prompts.
- Outcome: mean three-source vote entropy, defined exactly as in the original.
- Test: 2,000-resample bootstrap difference in means.
- Decision: support only when the 95% interval excludes zero above zero.
- `N_trials=1` for this follow-up.

The target of 64 is twice the pilot-based minimum produced by
`scripts/run_stage3_power.py`; the inflation reduces dependence on the noisy
effect estimate from the original 10 flagged examples.
