# Locked Natural-Video and Compositional Follow-up

Status: locked on 2026-09-03 before observing any follow-up outcomes. This is
an additive plan; it does not alter `RESEARCH_SEQUENCE_PLAN.md` or any earlier
threshold, result, or conclusion.

## Shared protocol

- Seeds: 11, 22, 33, 44, and 55.
- Learned comparisons use identical frozen inputs, splits, initialization
  families, and optimization budgets.
- Primary deltas report `N_trials=5` and paired 95% intervals from 10,000
  seed-bootstrap resamples. Per-example metrics report 10,000 paired bootstrap
  resamples where applicable.
- Every table is regenerated from prediction-level or trial-level artifacts.
- No test outcome may select a model, corruption severity, threshold, or
  acquisition weight.

## F1 — natural-video temporal and robustness evaluation

Use official UCF101 split 1. Fix these ten classes before download:
`Basketball`, `Biking`, `Diving`, `Drumming`, `HorseRiding`, `PlayingGuitar`,
`RockClimbingIndoor`, `Rowing`, `Skiing`, and `TaiChi`. Retain at most 70 train
and 30 test videos per class in lexicographic filename order. Sample 16 frames
uniformly from each video. Evaluate frozen CLIP class-prompt accuracy and
temporal embedding stability under frame reversal, deterministic 50% frame
drop, Gaussian blur radius 4, and brightness factor 0.25. Primary robustness
metric: clean-minus-corrupted top-1 accuracy. Natural-video conclusions are
limited to this fixed subset and official split.

## F2 — verified-negative retrieval

Train diagonal CLIP retrieval adapters on COCO training pairs. Compare random
negatives, the already-run nearest-neighbor hard-only policy, and a 50:50
mixture of random negatives plus *object-disjoint semi-hard negatives*.
Object-disjoint candidates must share no COCO instance category with the
positive image and are chosen from the top 10% of remaining frozen-CLIP
similarities. Primary metric: untouched COCO text-to-image Recall@1 delta,
mixed minus random. SugarCrepe is evaluation-only and is never used for
selection or fitting.

## F3 — temporal adapter

Freeze CLIP frame embeddings. Compare a mean-pooled linear classifier with a
temporal residual adapter consisting of a one-dimensional temporal convolution,
ReLU, residual mean pooling, and a linear class head. Train both for 30 epochs
with AdamW, learning rate 1e-3, batch size 32, using only UCF101 split-1 train.
Primary metric: clean top-1 accuracy delta, temporal minus mean-pooled.
Secondary: corrupted accuracy and worst-class recall.

## F4 — multi-objective active acquisition

At 10% and 20% correction budgets, compare random, disagreement-only, and a
fixed equal-weight composite of normalized disagreement, classifier uncertainty,
greedy embedding diversity, and low-quality membership. Correction uses COCO
silver labels for the three verifiable tasks only. Retrain the same frozen-CLIP
logistic head. Primary metric: macro test accuracy, composite minus
disagreement-only; secondary: ECE and bottom-blur-quartile accuracy.

## F5 — dedicated compositional evaluation

Evaluate frozen OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) on the official
SugarCrepe release using the existing COCO 2017 validation images. Report
positive-over-negative accuracy separately for add, replace, and swap
perturbation families, plus macro accuracy and paired bootstrap intervals.
This benchmark provides human-validated compositional negatives; it does not
provide a general safety certificate.

## F6 — reusable package

Move shared provenance, bootstrap, retrieval, acquisition, corruption, and
report-schema logic into importable modules under
`src/vision_language_classification/research/`. Scripts remain thin CLI
entrypoints. Add unit tests, dataset cards, reproduction commands, and a single
generated follow-up report.

## Stop conditions

- If official UCF101 or SugarCrepe cannot be acquired or its identity cannot be
  verified, stop that branch rather than substitute an unregistered dataset.
- If decoding failures exceed 5% in any class/split, report the failure and do
  not make comparative claims.
- Source videos, benchmark images, and model weights remain outside Git.
