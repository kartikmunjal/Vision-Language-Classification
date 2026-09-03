# Preregistration

Frozen before any Stage 3 outcome is computed.

## Hypothesis

Weak-label disagreement is higher for examples in the joint low-sharpness,
low-texture slice than outside that slice. This tests whether an aggressive
visual-quality gate preferentially removes examples for which text- and
image-derived supervision is least reliable.

The low-sharpness band is Laplacian variance below 40, matching the predecessor
Video-Curation pipeline's moderate filter boundary. The low-texture band is the
bottom quartile of grayscale local standard deviation, with the quartile cut
computed once on the training split only and then frozen for validation/test.

## Primary test

For each of the six binary tasks, compute three-source vote entropy. Compare the
per-example mean entropy inside versus outside the flagged slice using a
stratified bootstrap (2,000 resamples, stratified by source action category).
The primary aggregate is the macro-average difference across tasks, with its
95% percentile interval. This is one preregistered primary hypothesis
(`N_trials=1`). The hypothesis is supported only if the interval excludes zero
on the positive side.

## Active-correction test

At correction budgets of 5%, 10%, and 20%, compare correction in descending
vote-entropy order with equal-sized random correction. Each random baseline is
repeated 100 times using recorded seeds. Primary endpoint: paired change in
macro ECE on the untouched human-labeled evaluation set. Secondary endpoints:
macro accuracy and accuracy/ECE within the flagged slice. All estimates receive
95% bootstrap intervals and explicitly report `N_trials=3` budgets.

## Leakage controls

- Split by `source_video_id`, never by frame, before fitting or calibration.
- Derive weak labels and train only after split assignment.
- Fit temperature only on the calibration split.
- Human evaluation labels are never used for model fitting except in the
  explicitly simulated correction subset, which is drawn from training data.
- The final evaluation split remains untouched across correction budgets.

## Null-result policy

A null or reversed result is reported unchanged. Thresholds, tasks, primary
endpoint, and slice definition are locked after the first Stage 3 run.
