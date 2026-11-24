# Geometric Regularization with Optimal Transport for Zero-Shot Cross-Lingual Question Answering

Official implementation for the EACL 2027 paper:
**"Geometric Regularization with Optimal Transport for Zero-Shot Cross-Lingual Question Answering"**

**Authors:** Vinh Q. Vo<sup>1</sup>, Tin Ngo Trung<sup>1</sup>, Anh Trac Duc Dinh<sup>1,2</sup>, Tho T. Quan<sup>1*</sup>, Anh Duc Le<sup>3*</sup>
<sup>1</sup>Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT), VNU-HCM, Ho Chi Minh City, Vietnam
<sup>2</sup>Center for AI Research (CAIR), VinUniversity, Hanoi, Vietnam
<sup>3</sup>NTT Hi-Tech Institute, Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam
<sup>*</sup>Co-corresponding authors: [`qttho@hcmut.edu.vn`](mailto:qttho@hcmut.edu.vn), [`anhld@ntt.edu.vn`](mailto:anhld@ntt.edu.vn)

---

## Overview

Zero-shot cross-lingual extractive question answering (QA) transfers answer localization from English to target languages without target-language QA labels. However, standard representation alignment methods often suffer from two competing failure modes: **catastrophic forgetting** of source-language capabilities and **over-alignment** (where excessive global pulling blurs inter-token discriminability and degrades answer-boundary precision).

We show that global-only Optimal Transport (OT), without span-aware local projection or boundary regularization, can underperform a zero-shot baseline. We address this through a **Coordinated Alignment Framework**:

1. **Spatial Coordination (Macro OT + $\gamma$-Barycentric Span Projection):** Combines entropic Sinkhorn Optimal Transport ($\mathcal{L}_{ot}$) with span-aware local projection ($\mathcal{L}_{span}$) to transfer answer-boundary geometry without collapsing token separability.
2. **Domain Coordination (Frozen Teacher / LoRA Student):** Anchors representations via consistency regularization ($\mathcal{L}_{reg}$) against a frozen English teacher while adapting the student through LoRA layers, preserving English QA performance.
3. **Boundary Regularization in Logit Space:** Applies a margin-based boundary regularizer ($\mathcal{L}_{margin}$) directly in task logit space. Across multi-seed experiments, keeping this regularizer static outperforms or matches dynamic epoch scheduling, showing that boundary regularization's value lies in its presence rather than its schedule.

Our primary study uses **XLM-RoBERTa-base** (`xlm-roberta-base`) across **Vietnamese, Arabic, and Hindi** (with complete M1–M5 ablations on XQuAD and MLQA). We also include a supplementary single-seed reference run on **mmBERT** (`jhu-clsp/mmBERT-base`).

---

## Main Results (XLM-R, Seed 42)

Zero-shot cross-lingual QA performance (EM / F1) compared against baselines on XQuAD and MLQA:

| Target Language | Model | XQuAD (EM / F1) | MLQA (EM / F1) |
| :--- | :--- | :---: | :---: |
| **Vietnamese** | XLM-R Zero-shot | 46.22 / 63.64 | 46.05 / 67.21 |
| | Vanilla KD | 46.97 / 63.58 | **46.10** / 67.21 |
| | MAD-X (retrained) | 42.69 / 57.03 | 34.85 / 48.52 |
| | InfoXLM | 28.49 / 38.29 | 43.24 / 64.48 |
| | MINOTAUR | 41.26 / 58.05 | 41.42 / 61.79 |
| | **Ours (Coordinated, static)** | **49.70 / 69.73** | 46.00 / **67.60** |
| **Arabic** | XLM-R Zero-shot | 43.53 / 57.03 | 37.26 / 56.38 |
| | Vanilla KD | 43.95 / 57.13 | 37.22 / 56.12 |
| | MAD-X (retrained) | 32.35 / 45.79 | 21.44 / 33.89 |
| | InfoXLM | 23.45 / 30.96 | 34.47 / 53.71 |
| | MINOTAUR | 36.72 / 50.04 | 33.54 / 52.81 |
| | **Ours (Coordinated, static)** | **47.31 / 65.26** | **37.59 / 56.50** |
| **Hindi** | XLM-R Zero-shot | 44.54 / 60.21 | 42.88 / 60.40 |
| | Vanilla KD | 44.79 / 60.37 | 42.80 / 60.41 |
| | MAD-X (retrained) | 42.52 / 52.84 | 33.75 / 43.98 |
| | InfoXLM | 35.38 / 47.04 | 41.46 / 59.62 |
| | MINOTAUR | 29.75 / 42.78 | 36.93 / 53.41 |
| | **Ours (Coordinated, static)** | **52.94 / 66.94** | **45.87 / 61.69** |

Averaged across the three target languages, the coordinated framework improves EM / F1 over the zero-shot baseline by **+5.2 EM / +7.0 F1** on XQuAD and **+1.1 EM / +0.6 F1** on MLQA while keeping English retention close to the teacher model.

---

## Repository Structure

```text
.
├── train_stage1.py                 # Stage 1: Supervised English teacher training (SQuAD 2.0)
├── train_stage2.py                 # Stage 2: Teacher-student coordinated OT alignment (Vietnamese)
├── gpu_utils.py                    # DDP setup, auto GPU selection, and seeding utilities
├── squad_parallel_loader.py        # Parallel EN-VI data loader for Stage 2
├── check_data_overlap.py           # Context-level train/eval deduplication verification
├── mlqa_evaluation_v1.py           # Official MLQA evaluation script
├── download_prerequisites.py       # Helper to fetch pretrained models/datasets
│
├── phase1_dataloader/              # QA sample preprocessing & cross-lingual dataset builders
├── phase2_model/                   # Core dual-encoder LoRA/OT model wrappers (`model_core.py`)
├── phase3_loss/                    # Coordinated Sinkhorn OT, span projection, margin & reg losses (`losses.py`)
├── phase4-evaluation/              # Inference, JSON prediction pipeline, and diagnostic metrics
├── data/                           # XQuAD data loaders
├── losses/                         # Vanilla KD baseline loss
│
├── vietnamese/                     # Vietnamese evaluation scripts (`run_eval_all_vi.sh`)
├── arabic/                         # Arabic Stage-2 scripts, loaders, and eval (`run_ar.sh`, `run_ablation_ar.sh`)
├── hindi/                          # Hindi Stage-2 scripts, loaders, and eval (`run_hi.sh`, `run_ablation_hi.sh`)
├── script/                         # Slurm/shell runners for ablations, multi-seed runs, and exports
├── run_m3.py / run_m3.sh           # Multi-seed M3/M4/M5 evaluation & aggregation helpers
│
├── run_mmbert_pipeline.sh          # Supplementary single-seed mmBERT end-to-end reference pipeline
├── eval_and_aggregate_mmbert.py    # Evaluation & aggregation for supplementary mmBERT runs
├── Probe_mmBERT_Layers_Colab.ipynb # Layer probing notebook for mmBERT
│
├── baselines/                      # Retrained external baselines (MAD-X, InfoXLM, MINOTAUR)
├── analysis/                       # Transport entropy, fragmentation, and Hindi epoch diagnostics
├── benchmarks/                     # VRAM, parameter count, and throughput measurements
├── paper_tools/                    # Visualization and diagnostic plotting scripts for figures
│
├── backup_xlm_base/                # Preserved original XLM-RoBERTa-base reference source files
├── ACL_latex/                      # Camera-ready LaTeX manuscript, figures, and bibliography
├── Dockerfile                      # Container environment specification
├── requirements.txt                # Python package dependencies
└── LICENSE                         # MIT License
```

---

## Installation & Setup

### 1. Environment

Install Python dependencies (PyTorch with CUDA recommended):

```bash
pip install -r requirements.txt
```

A `Dockerfile` is also provided at the repository root for containerized execution.

### 2. Datasets & Contamination Check

Place the datasets under `dataset/`:
- **English Stage-1 & Retention:** `dataset/Squad2.0/` (`train-v2.0.json`, `dev-v2.0.json`)
- **Stage-2 Parallel Contexts:** `dataset/AIForge_vietnamese-squad/`, `dataset/ZIZOUArabic_Squad/`, `dataset/IndicSQuAD/`
- **Evaluation Benchmarks:** `dataset/xquad.{en,vi,ar,hi}.json` and `dataset/MLQA/test-context-*-question-*.json`

Verify zero context-level overlap between Stage-2 training corpora and XQuAD / MLQA evaluation sets:

```bash
python check_data_overlap.py
```

---

## Reproducing Experiments

### Stage 1: English Teacher Pretraining

Train the Stage-1 English QA teacher on SQuAD 2.0 (or use an existing `checkpoints/stage1_squad_best.pt`):

```bash
python train_stage1.py \
    --model_name xlm-roberta-base \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-5 \
    --seed 42
```

### Stage 2: Coordinated Alignment (XLM-R)

#### Vietnamese (Primary Branch)

Run the full coordinated configuration (**M5, static boundary regularizer**):

```bash
python train_stage2.py \
    --stage1_ckpt checkpoints/stage1_squad_best.pt \
    --model_name xlm-roberta-base \
    --batch_size 32 \
    --max_epochs 6 \
    --lambda_ot 0.5 \
    --lambda_span 1.0 \
    --lambda_margin 1.0 \
    --lambda_reg 50.0 \
    --lambda_qa 1.0 \
    --seed 42
```

Other ablation variants supported via CLI flags on `train_stage2.py`:
- **M4 (Dynamic Margin Schedule):** Add `--anneal_margin` to the M5 command above.
- **M3 (No Margin Regularizer):** Set `--lambda_margin 0.0`.
- **M2 (Global-Only OT):** Set `--lambda_span 0.0 --lambda_margin 0.0`.
- **M1 (Vanilla KD):** Set `--lambda_kd 1.0 --lambda_ot 0.0 --lambda_span 0.0 --lambda_margin 0.0`.

Multi-seed ablation and export scripts are located in `script/` (e.g., `script/run_ablation.sh`, `script/run_m4_m5_vi.sh`, `script/run_seeds.sh`).

#### Arabic and Hindi Branches

Run the end-to-end training & evaluation pipelines (Zero-shot, Vanilla KD, Static OT, Ours) for Arabic and Hindi:

```bash
bash arabic/run_ar.sh checkpoints/stage1_squad_best.pt
bash hindi/run_hi.sh checkpoints/stage1_squad_best.pt
```

### Evaluation, Diagnostics & Benchmarks

- **Language-level evaluations:** `vietnamese/run_eval_all_vi.sh`, `arabic/run_eval_all_ar.sh`, `hindi/run_eval_all_hi.sh`
- **Transport entropy & diagnostic scripts:** `bash analysis/run_hypothesis_tests.sh`
- **Efficiency & memory benchmarks:** `bash benchmarks/run_efficiency_all.sh`
- **Paper figures & diagnostics:** Scripts in `paper_tools/` (e.g., `paper_tools/make_figure2.py`, `paper_tools/make_summary_figure.py`)

### Supplementary Reference: mmBERT (Single Seed)

To run the exploratory single-seed (`seed=42`) check on `jhu-clsp/mmBERT-base`:

```bash
bash run_mmbert_pipeline.sh
```

---

## Citation

If you find this repository helpful in your research, please cite our EACL 2027 paper:

```bibtex
@inproceedings{vo-etal-2027-geometric,
    title = "Geometric Regularization with Optimal Transport for Zero-Shot Cross-Lingual Question Answering",
    author = "Vo, Vinh Q. and Ngo Trung, Tin and Dinh, Anh Trac Duc and Quan, Tho T. and Le, Anh Duc",
    booktitle = "Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics (EACL)",
    year = "2027"
}
```

## License

This project is released under the [MIT License](LICENSE).
