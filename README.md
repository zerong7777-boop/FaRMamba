# FaRMamba (ICONIP 2025)

Official PyTorch implementation of **FaRMamba**: *Frequency-based Learning and Reconstruction Aided Mamba for Medical Image Segmentation* (https://arxiv.org/abs/2507.20056) (accepted at **ICONIP 2025**).

---

## 1. Installation

### Dependencies
```bash
pip install -U numpy tqdm matplotlib timm einops
```

### Optional (depending on MSFM type)
```bash
# WT (wavelet)
pip install -U PyWavelets

# DCT
pip install -U scipy
```

---

## 2. Repository Layout (important)

Place `Network/` and `Module/` at the repository root so imports work:
```text
./
├── main_FaRMamba.py
├── loss_farmamba.py
├── Dataloader.py
├── common.py                 # REQUIRED by Dataloader.py (load_img, img2tensor)
├── Network/
│   ├── __init__.py
│   ├── FaRMamba.py
│   └── SwinUMamba.py
└── Module/
    ├── __init__.py
    ├── WT_stable.py
    ├── FFTtransformer.py
    ├── DCTtransformer.py
    ├── MSCA.py
    └── CBAM.py
```

> Note: `Dataloader.py` imports `load_img` / `img2tensor` from `common.py`. Please include `common.py` in this repo.

---

## 3. Data Preparation

The built-in dataset loader `PairedData` expects:
```text
DATASET_ROOT/
└── train/
    ├── image/   # images (same filenames as labels)
    └── label/   # masks
```

- Filenames must match between `image/` and `label/`.
- Masks should be **class-index** maps (values in `0..num_classes-1`).
  - If your masks are `0/255`, please convert them to `0/1` before training.

---

## 4. Train

### (A) Full FaRMamba (MSFM + SSRAE + LGRA + Fusion)
```bash
python main_FaRMamba.py \
  --exp_name farmamba_full \
  --dataset paireddata \
  --dataset_root /path/to/DATASET_ROOT \
  --num_classes 2 \
  --epochs 200 --batch_size 4 --lr 1e-4 \
  --pretrained \
  --use_msfm --msfm_type WT \
  --use_ssrae --share_mamba_backbone \
  --use_region_attention --region_attn_mode hard --region_attn_beta 4.0 \
  --use_fusion \
  --warmup_epochs 5 --warmup_mode sr_only --lambda_mode ema_ratio \
  --report_dice \
  --early_stop --early_stop_patience 30
```

### (B) Segmentation-only baseline
```bash
python main_FaRMamba.py \
  --exp_name baseline_seg \
  --dataset paireddata \
  --dataset_root /path/to/DATASET_ROOT \
  --num_classes 2 \
  --epochs 200 --batch_size 4 --lr 1e-4 \
  --pretrained \
  --report_dice
```

---

## 5. Outputs

Runs are saved to `--save_root` (default: `./runs_farmamba`) as:
```text
runs_farmamba/<exp_name>_<timestamp>/
├── config.json
├── train.log
├── metrics.csv
├── times.csv
├── figures/        # curves
└── checkpoints/    # best.pth / last.pth / (optional) epoch_*.pth
```

Validation is executed every `--eval_interval` epochs, and `best.pth` is selected by the **lowest validation loss**.

---

## 6. Reproducibility

- Use `--seed` (default: 42) for deterministic splits and training randomness.
- Results may vary slightly across different GPU types / driver stacks.
- For transformer/SSM-style backbones, enabling `--pretrained` is recommended.

---


## 7. References

This repository reuses / adapts implementations from the following projects (we thank the authors):
- **CBAM**: Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018.
- **MSCA**: Guo et al., *SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation*, NeurIPS 2022.  
  (MSCA module implementation follows open-source SegNeXt-style MSCA.)
- **SwinUMamba**: the SwinUMamba / U-Mamba style vision Mamba segmentation backbone implementation used as the `VSSMEncoder`.

If you believe any reference is missing or needs more explicit attribution (e.g., a specific GitHub repo link), please open an issue or contact the authors.

---

## 8. Citation

If you find this work useful, please cite:

```bibtex
@misc{rong2025farmambafrequencybasedlearningreconstruction,
      title={FaRMamba: Frequency-based learning and Reconstruction aided Mamba for Medical Segmentation}, 
      author={Ze Rong and ZiYue Zhao and Zhaoxin Wang and Lei Ma},
      year={2025},
      eprint={2507.20056},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.20056}, 
}
```

---
