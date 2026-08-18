# Explainability as a Diagnostic Tool for Multilingual Failure in Audio Deepfake Detection

> **Manuscript status:** [`research (7).pdf`](research%20(7).pdf) is an unpublished research draft. Its methods, analyses, results, and conclusions may change before submission or publication.

This repository contains the code and frozen experimental artifacts supporting the draft paper, *Explainability as a Diagnostic Tool for Multilingual Failure in Audio Deepfake Detection*. It studies why a wav2vec 2.0 XLS-R + AASIST detector degrades on controlled Hindi versus English Griffin-Lim copy-synthesis data, using occlusion, Integrated Gradients, faithfulness tests, silence analyses, and acoustic robustness controls.

The primary result reported in the draft is a higher EER for Hindi (30.49%) than English (16.18%) under the matched Common Voice / Griffin-Lim protocol. The analyses investigate the persistence of this gap after silence trimming and acoustic controls, and the associated shift in frequency-band attribution.

## Contents

| Area | Included material |
| --- | --- |
| Paper | `research (7).pdf` — current manuscript draft |
| Baseline and model support | `model.py`, `data_utils_SSL.py`, `RawBoost.py`, `eval_metric_LA.py`, `run_eval_2019LA.py`, and the pinned `fairseq-*` source |
| Phase 2 | English ASVspoof 2019 LA XAI replication: `phase2_xai_english_clean.py` and `phase2_outputs/` |
| Phase 3 | Silence-shortcut replication: `phase3_shortcut.py` and `phase3_backup-20260811T163152Z-1-001/` |
| Phase 4 | Hindi/English Griffin-Lim control-set preparation and quality checks |
| Phase 5 | Controlled English/Hindi evaluation, attribution analysis, and frozen results in `phase5_final_output_backup/` |
| Follow-up analyses | `exp1_*` through `exp8_*`, plus the continuation helpers for the reported robustness and explanation-drift analyses |

The tracked output directories contain the artifacts used by the draft, including attribution arrays, figures, score tables, run manifests, confidence-interval analyses, and statistical summaries. Raw datasets, audio clips, and model checkpoints are intentionally excluded.

## Requirements and external inputs

The research was run with Python 3.8.20, PyTorch 1.13.1, torchaudio 0.13.1, and an NVIDIA A100 GPU. The provided requirements list the analysis dependencies, while the local `fairseq-*` directory pins the XLS-R implementation revision used by `model.py`.

To reproduce model-dependent steps, obtain these external resources yourself and configure the paths in the relevant scripts:

- ASVspoof 2019 LA evaluation audio (the evaluation protocol is retained in `database/`)
- Mozilla Common Voice Hindi and English source data
- `xlsr2_300m.pt` (XLS-R 300M frontend)
- `best_SSL_model_LA.pth` (the pretrained countermeasure checkpoint)

Example environment setup:

```bash
conda create -n multilingual-deepfake python=3.8
conda activate multilingual-deepfake

# Install the appropriate PyTorch 1.13.1 / CUDA build for your system first.
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable .
cd ..
pip install -r requirements.txt
```

The model-dependent scripts expect `xlsr2_300m.pt` at the repository root and the countermeasure checkpoint at `pretrained_models/best_SSL_model_LA.pth` unless their configuration is changed.

## Research workflow

1. `run_eval_2019LA.py` verifies the in-domain ASVspoof 2019 LA baseline.
2. `phase2_xai_english_clean.py` reproduces the English occlusion / Integrated Gradients and faithfulness analysis.
3. `phase3_shortcut.py` tests the silence shortcut on ASVspoof 2019 LA.
4. `prepare_hindi_griffinlim.py` and `prepare_english_griffinlim.py` construct matched Common Voice Griffin-Lim control data; the corresponding `sanity_check_*.py` scripts validate it.
5. `phase5_hindi_eval.py` and `phase5b_occlusion_stats.py` run the controlled English–Hindi evaluation and explanation analysis.
6. `exp1_*`–`exp8_*` reproduce the paper's robustness, interaction, spectral, speaker-level, and frequency-intervention analyses from the frozen Phase 5 data.

Read the configuration blocks and docstrings in each script before execution: dataset locations, checkpoints, and hardware assumptions are environment-specific. The archived result directories allow the paper's reported analyses to be inspected without rerunning data preparation or GPU inference.

## Provenance and attribution

The detector implementation and pretrained-model workflow build on [Hemlata Tak et al.'s SSL_Anti-spoofing repository](https://github.com/TakHemlata/SSL_Anti-spoofing) and the associated wav2vec 2.0 XLS-R + AASIST work. The original MIT license is retained in [`LICENSE`](LICENSE). This repository's project-specific research scripts, controlled datasets, analyses, and frozen output artifacts support the draft paper above; it does not claim authorship of the underlying detector architecture.

If you use the underlying detector implementation, please cite the original work:

```bibtex
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```
