# CoSA: Consensus-based Saliency Agreement

## Overview

Official code for the paper **"CoSA: Consensus-based saliency agreement for annotation-free foundation model explainability
assessment validated in diabetic retinopathy"** (manuscript under review).

CoSA ranks image classifiers by how well their saliency maps agree with a **consensus map** built from the saliency
maps of several accurate foundation models, so that explanation quality can be compared **without lesion
annotations**. In the paper, seven vision foundation models were adapted with three strategies (classifier-only
training, partial fine-tuning, LoRA) for referable diabetic retinopathy (DR); the consensus map was validated
against expert pixel-level lesion annotations, and CoSA was computed on unannotated external images.

The training framework (`configs/`, `data/`, `models/`, `trainers/`, `scripts/run.sh`) follows the lab's
Opportunistic-FM code base.

---

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
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12 (NVIDIA PyTorch container 26.02). Training used 4 GPUs with DDP.

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
| `SNUH_*` | SNUH health-checkup cohort (only on the hospital server; see Data) |

## Data preparation

- Public datasets (see the paper's Data availability and references): EyePACS, MFIDDR, DDR, DRTiD (development);
  APTOS2019, Messidor-2, IDRiD, FGADR, DeepDRiD (external validation); IDRiD, FGADR and e-Ophtha pixel-level lesion
  masks (localization). Partitions are described in Supplementary Table S1.
- The SNUH health-checkup cohort cannot be shared (IRB); the SNUH steps below were run on the hospital server.
- No images, labels, predictions or weights are included in this repository.

## Model training, evaluation and explainability analysis

The table lists every step in order and the manuscript item it produces. Each script has its full command line
in its header.

| Step | Script | Manuscript item |
|---|---|---|
| 1. Train the 21 configurations | `scripts/run_dr.sh` (configs in `configs/referable_dr/`) | Table 2 models |
| 1b. Any-DR sensitivity | `scripts/run_dr.sh configs/any_dr/*.yml` | Supp. Table S4 |
| 2. Internal/external AUROC, sensitivity, specificity (bootstrap CIs) | `interpreters/bootstrap_classification_ci.py` | Table 2, Supp. Tables S2(b), S4 |
| 2b. Per-dataset external AUROC | `interpreters/per_dataset_external_auroc.py` | Supp. Table S2(c) |
| 2c. SNUH AUROC (hospital server) | `interpreters/snu_eval_dr.py` | Table 2, Supp. Tables S2(a), S4 |
| 2d. SNUH AUPRC | `analysis/snuh_auprc.py` | Supp. Table S2(a) |
| 2e. Prediction-level ensembles | `interpreters/ensemble_auroc.py` | Supp. Table S3 |
| 3. Saliency maps, annotated datasets (gradient-weighted attention rollout, 448 x 448) | `interpreters/snu_gen_saliency_attn.py --attn_mode grad_rollout` | Figure 2, Supp. Table S5 |
| 3b. Saliency maps, external images | `interpreters/gen_saliency_attn.py`; classifier-only configurations: `interpreters/gen_saliency_attn_multiscale_keys.py --scales 1.0` | Table 3 |
| 4. Per-case optimal Dice of individual maps | `interpreters/consensus_rules_v2.py` | Figure 2, Supp. Table S5 |
| 4b. Consensus map (L1 normalization, trimmed mean discarding 2 values at each end) and its Dice | `interpreters/consensus_rules_v5_means.py` (rule `UnitSum-Arith-T2x2`) | Figure 2, Supp. Tables S5, S7 |
| 4c. Reference rows (random, lesion-location, perfect score using saliency resolution) | `interpreters/baseline_avgmask_patchceiling.py` | Supp. Table S5 |
| 4d. Table with bootstrap CIs | `interpreters/supp_table_s5_dice.py` | Supp. Table S5 |
| 4e. Consensus vs individual configurations (paired, Holm) | `interpreters/consensus_rules_stats.py` | Results |
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

## Citation

To be updated after publication.
