# Follow-up Data Card

This file documents inputs; numerical results are generated separately.

## UCF101

- Source: University of Central Florida Center for Research in Computer Vision.
- Release: UCF101 archive plus official recognition train/test splits.
- Registered use: official split 1, ten fixed classes, at most 70 train and 30
  test videos per class, lexicographic selection, 16 uniformly sampled frames.
- Raw videos are not redistributed in this repository.
- Limitations: YouTube-derived historical benchmark, restricted class subset,
  no demographic labels, and possible compression/capture artifacts.

## COCO 2017

- Registered use: existing 5,000-image validation subset, original captions,
  and instance-category annotations for object-disjoint negative verification
  and annotation-free correction labels.
- COCO labels are described as silver labels rather than new human annotation.
- Raw images are not redistributed here.

## SugarCrepe

- Source: official RAIVNLab SugarCrepe release (NeurIPS 2023).
- Registered use: evaluation only; positive and human-validated negative
  captions across add, replace, and swap families.
- SugarCrepe uses COCO 2017 validation images, which are acquired separately.
- Benchmark annotations are not used to train or select the evaluated model.
