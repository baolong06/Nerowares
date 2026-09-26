---
license: other
language:
  - en
  - vi
tags:
  - eeg
  - motor-imagery
  - bci
  - bciciv-2a
pretty_name: BCICIV-2a EEG Motor Imagery
size_categories:
  - 100M<n<1G
---

# BCICIV-2a EEG Motor Imagery

## Provenance

This repository contains one consolidated CSV shard derived from the local
research copy identified as the Kaggle dataset
[`aymanmostafa11/eeg-motor-imagery-bciciv-2a`](https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a),
which describes the BCI Competition IV 2a motor-imagery recordings associated
with the Graz BCI laboratory. The uploader has represented that they have the
right to redistribute this research copy. The repository owner must preserve
any upstream attribution and terms that apply to the source dataset.

This card does not assert a SPDX license. `license: other` is intentional until
the upstream terms are independently mapped to a recognized license identifier.

## Contents

- `BCICIV_2a_all_patients.csv`
- 2,448 epochs from 9 subject identities
- 22 EEG channels per epoch
- 201 samples per epoch at 250 Hz
- labels: `left`, `right`, `foot`, `tongue`
- identity fields: `patient` (subject) and `epoch` (session boundary)

The CSV is a tabular, already-windowed representation. It must not be pooled
with Brennan, D01, Chisco, D09, or SparrKULee.

## Intended use

Research and reproducibility testing for the THINKING EEG pipeline, including
schema validation, checksum verification, leakage-safe subject/session splits,
and controlled training experiments.

## Limitations and safety

- This dataset is not a clinical diagnostic resource.
- The scientific gate for the current BCICIV-2a encoder is **false**; no
  deployability or scientific headline is claimed by this repository.
- Do not infer identity beyond the explicit `patient` and `epoch` fields.
- Do not use random epoch splits for scientific evaluation.
- Consumers must verify the immutable catalog revision and shard SHA-256 before
  training.
- The repository contains EEG-derived human-subject data. Users are responsible
  for complying with the source terms, applicable law, institutional review
  requirements, and any restrictions imposed by the data custodian.

## Integrity

Measured source shard:

- bytes: `214848010`
- SHA-256: `fa52c442278b63c9add50939b453f0668719ff1137dd596b5b1cac90fe3aa6cd`

The GitHub catalog must pin the actual 40-character Hugging Face commit SHA;
`main` and other moving revisions are not valid for scientific runs.

## Citation and attribution

Cite the original BCI Competition IV 2a dataset and the upstream Kaggle source
when using this derived tabular copy. Do not represent this repository as the
original competition distribution.

## Release status

`gate_pass: false` for the current scientific evaluation. This publication
provides versioned data access only; it does not certify model quality or
clinical/deployment readiness.
