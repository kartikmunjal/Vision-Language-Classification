# Locked Six-Sequence Research Plan

Status: locked before running the six-sequence extension on 2026-09-03.

This extension connects the existing audio-curation, video-curation,
video-generation, video-reward, and vision-language projects. Existing locked
thresholds and previous results are inputs and are never retuned here.

## Shared rules

- Primary units, source paths, file hashes, seeds, package versions, and exact
  arguments are recorded in every result artifact.
- Seeds are 11, 22, 33, 44, and 55 for learned comparisons.
- Confidence intervals use 10,000 bootstrap resamples unless runtime makes that
  immaterial for a deterministic preprocessing statistic.
- A learned method is called better only when the paired 95% confidence
  interval for its prespecified primary delta excludes zero.
- COCO instance annotations are called *silver labels*, not human judgments.
- Synthetic image sequences test controlled temporal robustness; they are not
  represented as natural-video performance.
- Documentation numbers are rendered from result JSON by a named script.

## Sequence 1 — unified data-quality benchmark

Normalize committed evidence from Audio-Data-Creation,
Video-Quality-Reward-Modeling, and Vision-Language-Classification into a shared
schema. Preserve each source's estimand and uncertainty; do not pool unlike
metrics into a single headline score. The output is a cross-modal evidence
table and provenance manifest.

## Sequence 2 — learned multimodal reward

Use the 200 genuine video preference pairs and their four frozen metric
differences. Exclude ties from binary fitting. Compare a logistic Bradley-Terry
model with the repository's frozen handcrafted composite using five-fold
out-of-fold predictions, with folds grouped by normalized prompt text.
Primary metric: pairwise accuracy delta (learned minus handcrafted). Secondary:
ROC-AUC and Brier score. Report pair-bootstrap 95% intervals and `N_trials=5`.

## Sequence 3 — temporal vision-language evaluation

Construct eight-frame sequences from untouched COCO test images using four
deterministic trajectories: static, horizontal translation, progressive blur,
and progressive darkening. Evaluate frozen CLIP image-text ranking consistency
and embedding drift. Primary metric: correct-caption rank-1 retention from
frame 0 to frame 7, by trajectory. This is a controlled robustness benchmark.

## Sequence 4 — active correction with downstream retraining

For the three COCO-verifiable tasks (`human_present`, `animal_present`, and
`multiple_subjects`), replace weak ensemble training labels using COCO silver
labels at 10% and 20% budgets. Compare highest-disagreement targeting with an
equal-size random correction under paired seeds. Train the same classifier in
all arms. Primary metric: macro test accuracy delta, targeted minus random;
secondary: ECE and worst quality-slice accuracy. Report `N_trials=5` and paired
95% intervals. The pre-existing label-only simulation remains diagnostic.

## Sequence 5 — retrieval and hard-negative mining

Freeze CLIP image/text embeddings for the COCO subset. Train equal-capacity
linear projection adapters with in-batch random negatives or mined hard
negatives, paired by seed and optimization budget. Primary metric: text-to-image
Recall@1 delta on the untouched test split; secondary: Recall@5, image-to-text
retrieval, median rank, and quality slices. Report `N_trials=5` and paired 95%
intervals.

## Sequence 6 — multimodal safety and robustness

Evaluate the frozen baseline and best eligible learned components under blur,
darkening, occlusion, caption negation, and caption subject-count swaps.
Prespecified outputs are clean-to-corrupted performance deltas, prediction flip
rates, confidence shifts, and worst-slice metrics. This is a robustness audit,
not a comprehensive safety certification. No robustness intervention is tuned
on the test outcomes.

## Scope boundary

No source video files or trained CogVideoX checkpoints are present in the
provided repositories. Consequently, this plan does not claim generative-video
retraining or natural-video temporal generalization. Those require separately
licensed media/checkpoints and a new prospective amendment.
