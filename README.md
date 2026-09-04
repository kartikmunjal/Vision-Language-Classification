# Vision-Language Classification

An auditable weak-supervision pipeline for six visual attributes. It extends
the frame captions and quality metadata produced by
`Video-Curation`, then asks whether caption rules, a text-only LLM, and an
image-only CLIP labeler fail on the same quality-sensitive slice.

Author: Kartik Munjal

## Research design

The fixed tasks are `multiple_subjects`, `outdoor`, `human_present`,
`animal_present`, `dynamic_scene`, and `night`. Each is binary and is modeled
with its own logit. The project deliberately distinguishes three things:

1. Weak-label quality: agreement and Cohen's kappa against 180 human-reviewed
   examples, each with a 95% bootstrap interval.
2. Classifier quality: accuracy and calibration against human labels, not
   merely agreement with its training targets.
3. The preregistered finding: whether three-source disagreement concentrates
   in the low-sharpness, low-texture slice, and whether entropy-targeted human
   correction beats equally sized random correction.

No result numbers are checked into this README. Reports and figures are written
under `results/` by named scripts, preventing narrative numbers from drifting
away from primary data. See [PREREGISTRATION.md](PREREGISTRATION.md) before the
first Stage 3 run.

## Input contract

Use the enriched JSONL written by Video-Curation's BLIP-2 multitask annotation
stage. Required fields are `image_path` (or supply an extracted middle-frame
path), `caption`, and a stable video identifier via `source_video_id`,
`video_id`, `source_path`, or `path`. The adapter preserves `label` as
`source_category`, plus `blur_score`, `texture_score`, and `split` when present.

Fallback captions such as "a person performing ..." are rejected because they
leak the source action label and are not BLIP-2 observations. Splits are assigned
by source video, so frames from the same clip cannot cross train, calibration,
and test boundaries.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,plots]'
pytest -q

# Standalone fallback when the Video-Curation manifest is unavailable
python scripts/prepare_coco2017.py data/raw/coco2017 data/processed/manifest.jsonl

python scripts/prepare_manifest.py data/raw/enriched.jsonl data/processed/manifest.jsonl
python scripts/run_rule_labeler.py data/processed/manifest.jsonl data/processed/rules.jsonl
python scripts/make_human_validation.py data/processed/manifest.jsonl data/annotations/human_validation.csv
```

Complete every task cell in the CSV with `0` or `1`, without consulting weak
labels. Then measure Stage 1:

```bash
python scripts/evaluate_weak_labels.py \
  data/annotations/human_validation.csv data/processed/rules.jsonl \
  results/stage1_rule_agreement.json
```

For Stage 2, produce LLM outputs in the same JSONL contract as the rule output,
and run the image-only labeler:

```bash
.venv/bin/pip install -e '.[train]'
python scripts/run_clip_labeler.py data/processed/manifest.jsonl data/processed/clip.jsonl --device cuda
python scripts/run_hf_llm_labeler.py data/processed/manifest.jsonl data/processed/llm.jsonl --device cuda
python scripts/ensemble_labels.py \
  data/processed/rules.jsonl data/processed/llm.jsonl data/processed/clip.jsonl \
  data/processed/ensemble.jsonl

python scripts/train_classifier.py \
  data/processed/manifest.jsonl data/processed/ensemble.jsonl \
  checkpoints/baseline --device cuda
```

The LLM interface is provider-neutral: `label_with_llm` accepts a completion
callable, validates strict structured output, and requires callers to persist
the provider/model/version. This keeps API choice and credentials outside the
research logic.

Run the locked Stage 3 hypothesis only after the manifest contains texture
scores and all three sources are complete:

```bash
python scripts/run_stage3_hypothesis.py \
  data/processed/manifest.jsonl data/processed/ensemble.jsonl \
  results/stage3_preregistered_hypothesis.json
```

## GPU execution

Training and CLIP labeling can run on the Windows RTX 3070 host. Clone the same
repository there, install the `train` extra in a virtual environment, copy only
manifests and referenced frames, and invoke scripts with `--device cuda`.
Artifacts must record the commit hash, model identifier, seed, configuration,
and input-manifest SHA-256 before they are accepted into `results/`.

## Honest status

The annotation-free COCO experiment has been run end to end. Generated reports
under `results/` contain all empirical values and uncertainty intervals. Three
tasks have independent silver ground truth from COCO's original human instance
annotations; the remaining three have source-agreement evidence only. A new
six-task manual review would strengthen the study but is not represented as
having occurred.

### Annotation-free evaluation

When new manual review is unavailable, `build_coco_silver_labels.py` derives
independent silver labels from COCO's original human instance annotations for
human presence, animal presence, and multiple salient people/animals. The
annotation-free report makes ground-truth claims only for those three tasks.
Outdoor, dynamic-scene, and night results remain source-agreement diagnostics.

## Six-sequence multimodal extension

The prospectively locked [research sequence](RESEARCH_SEQUENCE_PLAN.md) has now
been executed across six connected studies: a cross-modal evidence schema, a
human-preference reward model, controlled temporal evaluation, active
correction with actual downstream retraining, hard-negative retrieval, and a
multimodal robustness audit. The generated [full report](results/research_sequence/REPORT.md)
contains outcomes and confidence intervals. The hard-negative experiment is a
reported failure, not a tuned-away result. Regenerate every table with:

```bash
python scripts/run_sequences_1_2.py --abc-root ../ --output-dir results/research_sequence
python scripts/extract_research_embeddings.py \
  --manifest data/processed/manifest.jsonl \
  --image-root data/raw/coco2017/val2017 \
  --output data/processed/research_embeddings.npz --device cuda
python scripts/analyze_sequences_3_6.py \
  --embeddings data/processed/research_embeddings.npz \
  --processed-dir data/processed --output-dir results/research_sequence
python scripts/render_research_sequence.py
```

## Natural-video follow-up

The additive [follow-up plan](FOLLOWUP_RESEARCH_PLAN.md) fixes an official
UCF101 split-1 subset, a frozen-CLIP temporal adapter, object-disjoint retrieval
negatives, multi-objective active acquisition, and official SugarCrepe
evaluation. Results are generated into `results/followup/`; raw benchmark media
and frozen embedding caches remain outside Git.

```bash
# CPU follow-ups over the existing COCO artifacts
PYTHONPATH=src python scripts/run_coco_followups.py \
  --embeddings data/processed/research_embeddings.npz \
  --processed-dir data/processed --output-dir results/followup

# GPU entrypoints (see each --help for registered paths)
python scripts/run_natural_video_followup.py --help
python scripts/run_sugarcrepe.py --help
python scripts/render_followup_report.py
```
