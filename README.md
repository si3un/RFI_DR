# CoSA: Consensus-based Saliency Agreement

Code for the paper **"CoSA: Consensus-based saliency agreement for annotation-free foundation model explainability
assessment validated in diabetic retinopathy"** (manuscript under review).

## Overview

CoSA ranks image classifiers by how well their saliency maps agree with a **consensus map** built from the saliency
maps of several accurate foundation models, so that explanation quality can be compared **without lesion
annotations**. In the paper, seven vision foundation models were adapted with three strategies (classifier-only
training, partial fine-tuning, LoRA) for referable diabetic retinopathy (DR); the consensus map was validated
against expert pixel-level lesion annotations, and CoSA was computed on unannotated external images.

Contents: [System requirements](#system-requirements) · [Installation](#installation) · [Demo](#demo) ·
[Reproducing the paper](#reproducing-the-paper) · [Repository layout](#repository-layout) · [Data](#data) ·
[Citation](#citation)

## System requirements

- **Operating system:** Linux. Experiments were run in the NVIDIA PyTorch container 26.02 (Python 3.12,
  torch 2.11, torchvision 0.25).
- **Hardware:** model training and saliency generation used NVIDIA A100 GPUs (4 GPUs with DistributedDataParallel
  for training). The demo and the `analysis/` scripts run on a standard CPU.
- **Software dependencies:** listed with tested versions in `requirements.txt`.

## Installation

```bash
git clone https://github.com/si3un/RFI_DR.git
cd RFI_DR
pip install -r requirements.txt
```

Install the `torch`/`torchvision` build that matches your CUDA version (https://pytorch.org). Installing the
dependencies with pip typically takes a few minutes.

## Demo

`demo/cosa_demo.py` runs the core of the method on synthetic saliency maps, so no images, models or annotations are
needed. It uses the same consensus and CoSA functions as the paper's analysis
(`interpreters/ext_consensus_similarity_loo.py`):

```bash
python demo/cosa_demo.py
```

It builds the consensus map of 14 synthetic configuration maps (equation 1 of the paper), computes CoSA for every
map, scores lesion localization with the per-case optimal Dice, and writes `demo/cosa_demo.png`. Expected output
(runs in a few seconds on a CPU):

```
Rank correlation with Dice: CoSA rho 0.60, CoSA top-k 0.96
Consensus Dice 0.830 vs best individual 0.826
```

## Reproducing the paper

### Paths (environment variables)

No path is hard-coded; configs and scripts read these variables.

| Variable | Meaning |
|---|---|
| `DETECTION_CSV` | development-cohort CSV (`image_path`, `label`, `grade`, boolean `train`/`valid`/`test` columns) |
| `EXTERNAL_CSV` | external-validation CSV (`image_path`, `label`, `grade`, `dataset`) |
| `PRETRAINED_ROOT` | folder with the public pretrained backbone weights |
| `DINOV3_REPO` | local clone of the DINOv3 repository (`hubconf.py`), for the DINOv3 backbones |
| `SAVE_ROOT` | where checkpoints are written |
| `OUTPUT_ROOT` | where analysis outputs are written |
| `EOPHTHA_ROOT`, `IDRID_SEG_ROOT`, `FGADR_SEG_ROOT` | annotated datasets used for lesion localization |
| `AUTOMORPH_M0_DIR`, `AUTOMORPH_MAP_CSV` | AutoMorph-preprocessed (M0) external images used for saliency on unannotated images |
| `SNUH_*` | SNUH health-checkup cohort (only inside the hospital environment; see [Data](#data)) |

### Training

`scripts/run_dr.sh` trains all 21 configurations (`configs/referable_dr/`, 7 backbones x 3 adaptation strategies)
with `scripts/run.sh`. The training code is built for efficient runs on one or several GPUs:

- **DistributedDataParallel** (one process per GPU, launched with `torchrun`); a single process is used
  automatically when only one GPU is visible.
- **Automatic mixed precision**, with bf16 or fp16 chosen from the hardware (`amp_dtype: auto`).
- **Cached backbone features for classifier-only training** (`ft.method: fast_linear`): the frozen backbone is run
  once and only the head is trained afterwards.
- **Early stopping** on the validation AUROC, and a complete record of every run (config, code snapshot, installed
  package versions, metrics, test predictions and an HTML report) in `$SAVE_ROOT/<run name>_<timestamp>/`.

```bash
export SAVE_ROOT=... PRETRAINED_ROOT=... DINOV3_REPO=... DETECTION_CSV=... EXTERNAL_CSV=...
bash scripts/run_dr.sh                               # all 21 configurations
bash scripts/run_dr.sh configs/referable_dr/dinov2_vitl14_lora.yml   # a single configuration
```

### Evaluation and explainability analysis

The table lists every step in order and the manuscript item it produces. Each script has its full command line in
its header.

| Step | Script | Manuscript item |
|---|---|---|
| 1. Train the 21 configurations | `scripts/run_dr.sh` (configs in `configs/referable_dr/`) | Table 2 models |
| 1b. Any-DR sensitivity | `scripts/run_dr.sh configs/any_dr/*.yml` | Supp. Table S4 |
| 2. Internal/external AUROC, sensitivity, specificity (bootstrap CIs) | `interpreters/bootstrap_classification_ci.py` | Table 2, Supp. Tables S2(b), S4 |
| 2b. Per-dataset external AUROC | `interpreters/per_dataset_external_auroc.py` | Supp. Table S2(c) |
| 2c. SNUH AUROC (hospital environment) | `interpreters/snu_eval_dr.py` | Table 2, Supp. Tables S2(a), S4 |
| 2d. SNUH AUPRC | `analysis/snuh_auprc.py` | Supp. Table S2(a) |
| 2e. Prediction-level ensembles | `interpreters/ensemble_auroc.py` | Supp. Table S3 |
| 3. Saliency maps, annotated datasets (gradient-weighted attention rollout, 448 x 448) | `interpreters/snu_gen_saliency_attn.py --attn_mode grad_rollout` | Figure 2, Supp. Table S5 |
| 3b. Saliency maps, external images | `interpreters/gen_saliency_attn.py`; classifier-only configurations: `interpreters/gen_saliency_attn_multiscale_keys.py --scales 1.0` | Table 3 |
| 4. Per-case optimal Dice of individual maps | `interpreters/eval_individual_localization.py` | Figure 2, Supp. Table S5 |
| 4b. Consensus map (L1 normalization, trimmed mean discarding 2 values at each end) and its Dice | `interpreters/eval_consensus_localization.py` (rule `UnitSum-Arith-T2x2`) | Figure 2, Supp. Tables S5, S7 |
| 4c. Reference rows (random, lesion-location, perfect score using saliency resolution) | `interpreters/baseline_avgmask_patchceiling.py` | Supp. Table S5 |
| 4d. Table with bootstrap CIs | `interpreters/supp_table_s5_dice.py` | Supp. Table S5 |
| 4e. Consensus vs individual configurations (paired, Holm) | `interpreters/compare_consensus_vs_individual.py` | Results |
| 4f. Normalization x trimming comparison | `analysis/supp_s7_norm_trim.py` | Supp. Table S7 |
| 5. CoSA on 2,617 unannotated external images (full, self- and lineage-leave-one-out) | `interpreters/ext_consensus_similarity_loo.py --rule arith_trim2` | Table 3, Supp. Table S6 |
| 5b. Trimming one value at each end | same script, `--rule arith_trim1`; `interpreters/table3_compare.py` | Supp. Table S6 |
| 5c. Pairwise comparisons | `interpreters/table3_pairwise.py` | Results |
| 5d. Correlations of CoSA and AUROC with Dice; leave-one-out summaries | `analysis/cosa_correlations.py` | Results, Supp. Table S6 |
| 6. Model selection and its threshold sensitivity | `analysis/selection_rule.py` | Results, Table 3 |
| 7. Figures | `figures/make_figure1.py`, `figures/make_figure2.py`, `interpreters/figure3_maps.py` (Figure 3, Supp. Figure S2) | Figures 1-3, Supp. Figure S2 |

Example (Table 3):

```bash
python interpreters/ext_consensus_similarity_loo.py --rule arith_trim2 --targets pool \
    --datasets aptos deepdrid messidor2 --auc_min 0.92 --sp_size 448 \
    --sal_root $OUTPUT_ROOT/saliency_attn --cons_sal_root $OUTPUT_ROOT/saliency_attn \
    --model_roots $SAVE_ROOT/dr_clean $SAVE_ROOT/ckpts_aux \
    --out_dir $OUTPUT_ROOT/ext_cons_sim_1x14_arith_trim2
```

Expected run time: CoSA at full resolution (`--sp_size 448`) takes hours (minutes at `--sp_size 56`); the
`analysis/` scripts run in seconds.

## Repository layout

```
configs/
  referable_dr/   21 run configurations (7 backbones x 3 adaptation strategies) exactly as used
  any_dr/         the same 21 configurations for the any-DR sensitivity analysis (Supplementary Table S4)
data/             CSV-driven dataset
models/           backbone builders (DINOv2, DINOv3, RETFound, RETFound-DINOv2, MAE, OpenCLIP), LoRA, heads
trainers/         training loop (DDP/AMP), metrics, run records
scripts/          run.sh (one config), run_dr.sh (all configs), weight download / CSV helpers
interpreters/     saliency maps, consensus map, lesion localization, CoSA, classification statistics
analysis/         small scripts that recompute numbers from saved result files (no images or models needed)
figures/          Figure 1 (schematic) and Figure 2
demo/             CoSA demo on synthetic saliency maps
```

## Data

- Public datasets (see the paper's Data availability and references): EyePACS, MFIDDR, DDR, DRTiD (development);
  APTOS2019, Messidor-2, IDRiD, FGADR, DeepDRiD (external validation); IDRiD, FGADR and e-Ophtha pixel-level lesion
  masks (localization). Partitions are described in Supplementary Table S1.
- The SNUH health-checkup cohort cannot be shared (IRB); the SNUH steps were run inside the hospital environment.
- No images, labels, predictions or weights are included in this repository.

## Citation

To be updated after publication.
