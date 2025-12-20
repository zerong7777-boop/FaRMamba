# main_FaRMamba.py
# -------------------------------------------------------------
# A clean, user-friendly training script for FaRMamba,
#
# Features:
#   - Dataset switching interface (built-in PairedData + dynamic import)
#   - Model selection interface (FaRMamba + compatible legacy tuple outputs)
#   - Ablation switches (MSFM/SSRAE/LGRA/Fusion/Shared-Mamba + degrade params)
#   - Early stopping
#   - Time logging (per-epoch + total)
#   - Rich console + file logging
#   - CSV metrics + plots + checkpoints (best/last/periodic)
#
# Run example:
#   python main_FaRMamba.py \
#     --exp_name farmamba_covid \
#     --dataset_root /path/to/dataset \
#     --dataset paireddata \
#     --num_classes 2 \
#     --epochs 200 --batch_size 4 --lr 1e-4 \
#     --use_msfm --msfm_type WT \
#     --use_ssrae --share_mamba_backbone --use_region_attention \
#     --warmup_epochs 5 --lambda_mode ema_ratio \
#     --early_stop_patience 30
# -------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"  # 强制无GUI后端，避免Tk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
import json
import time
import math
import argparse
import random
import datetime
from dataclasses import asdict
from typing import Any, Dict, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

# Optional plotting
import matplotlib.pyplot as plt


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def human_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(os.path.dirname(log_path))
        self.f = open(log_path, "a", encoding="utf-8")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def log(self, msg: str, also_print: bool = True):
        if also_print:
            print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, v: float, n: int = 1):
        self.sum += float(v) * int(n)
        self.cnt += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path: str, header: Tuple[str, ...], row: Tuple[Any, ...]) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def plot_curves(save_dir: str, history: Dict[str, list]) -> None:
    ensure_dir(save_dir)
    # Plot each metric that looks like a curve
    for k, v in history.items():
        if not isinstance(v, list) or len(v) == 0:
            continue
        if not all(isinstance(x, (int, float, np.number)) for x in v):
            continue
        plt.figure()
        plt.plot(range(1, len(v) + 1), v, marker="o")
        plt.title(k)
        plt.xlabel("Epoch")
        plt.ylabel(k)
        plt.grid(True)
        out = os.path.join(save_dir, f"{k.replace('/', '_')}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def try_import(qualified_name: str):
    """
    Dynamically import a class/function from "module.submodule:ClassName" or "module.submodule.ClassName".
    """
    if ":" in qualified_name:
        mod_name, attr = qualified_name.split(":", 1)
    else:
        parts = qualified_name.split(".")
        mod_name, attr = ".".join(parts[:-1]), parts[-1]
    mod = __import__(mod_name, fromlist=[attr])
    return getattr(mod, attr)


def ensure_4d_image(x: torch.Tensor) -> torch.Tensor:
    # Accept [B,H,W] or [B,1,H,W] or [B,C,H,W]
    # print(x.shape)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Invalid image shape: {tuple(x.shape)}")


def ensure_label_shape(y: torch.Tensor) -> torch.Tensor:
    """
    Accept labels with shapes:
      - [B,H,W]
      - [B,1,H,W]
      - [B,C,H,W]
      - [B,1,1,H,W]  (your current case)
      - [B,*,H,W] where extra dims are 1 (will be squeezed)
    Return: [B,H,W] (long)
    """
    # squeeze all singleton dims except batch
    if y.dim() >= 4:
        # keep dim0 (batch), squeeze other singleton dims
        squeeze_dims = [d for d in range(1, y.dim() - 2) if y.size(d) == 1]
        if len(squeeze_dims) > 0:
            y = y.squeeze(dim=squeeze_dims[0])
            for d in squeeze_dims[1:]:
                y = y.squeeze(dim=d-1)  # dims shift after squeezing

    # After squeeze, handle common cases
    if y.dim() == 5:
        # still 5D -> force squeeze all singleton dims between batch and HW
        y = y.squeeze(1).squeeze(1)

    if y.dim() == 4 and y.shape[1] == 1:
        y = y[:, 0]              # [B,H,W]
    elif y.dim() == 4 and y.shape[1] > 1:
        y = y.argmax(dim=1)      # [B,H,W]
    elif y.dim() == 3:
        pass
    else:
        raise ValueError(f"Invalid label shape after squeeze: {tuple(y.shape)}")

    return y



@torch.no_grad()
def mean_dice_from_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int,
                          ignore_index: Optional[int] = None, include_background: bool = False) -> float:
    """
    Compute mean Dice on argmax predictions. For reporting only (non-differentiable).
    logits: [B,C,H,W], labels: [B,H,W]
    """
    pred = logits.argmax(dim=1)  # [B,H,W]
    gt = labels.long()

    dices = []
    class_ids = list(range(num_classes))
    if not include_background and num_classes > 1:
        class_ids = class_ids[1:]

    for c in class_ids:
        if ignore_index is not None:
            valid = gt != ignore_index
            p = (pred == c) & valid
            g = (gt == c) & valid
        else:
            p = (pred == c)
            g = (gt == c)

        inter = (p & g).sum().item()
        denom = p.sum().item() + g.sum().item()
        if denom == 0:
            # If class absent in both, define dice=1 (common convention) to avoid penalizing
            dices.append(1.0)
        else:
            dices.append((2.0 * inter) / (denom + 1e-8))

    if len(dices) == 0:
        return 0.0
    return float(np.mean(dices))


# -------------------------
# Dataset factory
# -------------------------
def build_dataset(args, logger: Logger):
    """
    Two ways:
      (1) --dataset paireddata  (uses Dataloader.PairedData like your old script)
      (2) --dataset_cls "some.module:DatasetClass" for custom datasets
    """
    if args.dataset_cls:
        ds_cls = try_import(args.dataset_cls)
        logger.log(f"[DATA] Using dataset_cls={args.dataset_cls}")
        ds = ds_cls(**json.loads(args.dataset_kwargs))
        return ds

    # Built-in option matching your old code
    if args.dataset.lower() == "paireddata":
        try:
            from Dataloader import PairedData
        except Exception as e:
            raise ImportError(
                "Failed to import PairedData from Dataloader. "
                "Either fix your PYTHONPATH or use --dataset_cls to provide a dataset class."
            ) from e

        logger.log(f"[DATA] Using PairedData(root={args.dataset_root}, target={args.train_target}, use_num={args.use_num})")
        ds = PairedData(root=args.dataset_root, target=args.train_target, use_num=args.use_num)
        return ds

    raise ValueError(f"Unknown dataset '{args.dataset}'. Use 'paireddata' or provide --dataset_cls.")


# -------------------------
# Model factory (FaRMamba + compatible legacy output adapters)
# -------------------------
def build_model(args, device, logger: Logger) -> nn.Module:
    name = args.model.lower()

    if name in {"farmamba", "sota"}:
        # Ensure local import works when running from elsewhere
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        try:
            from Network.FaRMamba import FaRMamba  # your paper-aligned file
        except Exception as e:
            raise ImportError("Failed to import FaRMamba from FaRMamba.py. Check file path/name.") from e

        model = FaRMamba(
            num_classes=args.num_classes,
            in_chans=args.in_chans,
            pretrained=args.pretrained,

            # MSFM
            use_msfm=args.use_msfm,
            msfm_type=args.msfm_type,

            # SSRAE / shared mamba / LGRA
            use_ssrae=args.use_ssrae,
            share_mamba_backbone=args.share_mamba_backbone,
            use_region_attention=args.use_region_attention,
            region_attn_mode=args.region_attn_mode,
            region_attn_beta=args.region_attn_beta,

            # fusion
            use_fusion=args.use_fusion,

            # degrade
            degrade_scale_factor=args.degrade_scale_factor,
            degrade_blur_kernel=args.degrade_blur_kernel,
            degrade_blur_sigma=args.degrade_blur_sigma,
            degrade_noise_std=args.degrade_noise_std,
            degrade_noise_train_only=args.degrade_noise_train_only,
            degrade_blur_learnable=args.degrade_blur_learnable,
        ).to(device)

        logger.log("[MODEL] Built FaRMamba with ablation switches:")
        logger.log(f"        use_msfm={args.use_msfm}, msfm_type={args.msfm_type}")
        logger.log(f"        use_ssrae={args.use_ssrae}, share_mamba_backbone={args.share_mamba_backbone}")
        logger.log(f"        use_region_attention={args.use_region_attention}, mode={args.region_attn_mode}, beta={args.region_attn_beta}")
        logger.log(f"        use_fusion={args.use_fusion}")
        return model

    # Generic import for other models (optional)
    if args.model_cls:
        cls = try_import(args.model_cls)
        logger.log(f"[MODEL] Using model_cls={args.model_cls} kwargs={args.model_kwargs}")
        model = cls(**json.loads(args.model_kwargs)).to(device)
        return model

    raise ValueError(f"Unsupported model '{args.model}'. Use --model farmamba or provide --model_cls.")


def forward_with_aux(model: nn.Module, images: torch.Tensor, labels: Optional[torch.Tensor], need_aux: bool):
    """
    Unify various model outputs into:
      seg_logits, aux_dict
    Supports:
      - FaRMamba: returns seg or (seg, aux)
      - Legacy: returns (seg, sr_pred_list, sr_tar_list) or similar
    """
    # Prefer FaRMamba signature: model(x, labels=..., return_aux=...)
    try:
        out = model(images, labels=labels, return_aux=need_aux)
    except TypeError:
        # Fallback: old signature model(x, labels)
        out = model(images, labels)

    # Case 1: (seg, aux_dict)
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], dict):
        seg, aux = out
        return seg, aux

    # Case 2: (seg, sr_pred, sr_tar) (legacy)
    if isinstance(out, (tuple, list)) and len(out) == 3:
        seg, sr_pred, sr_tar = out
        # take last if list/tuple
        if isinstance(sr_pred, (tuple, list)):
            sr_pred = sr_pred[-1]
        if isinstance(sr_tar, (tuple, list)):
            sr_tar = sr_tar[-1]
        aux = {"sr_pred": sr_pred, "sr_target": sr_tar}
        return seg, aux

    # Case 3: only seg
    return out, {}


# -------------------------
# Checkpointing
# -------------------------
def save_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: Optional[Any],
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    epoch: int,
                    best_val: float,
                    extra: Optional[Dict[str, Any]] = None):
    ckpt = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    ensure_dir(os.path.dirname(path))
    torch.save(ckpt, path)


def load_checkpoint(path: str,
                    model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    map_location: str = "cpu") -> Tuple[int, float, Dict[str, Any]]:
    if not os.path.exists(path):
        return 0, float("inf"), {}
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    extra = ckpt.get("extra", {}) or {}
    return start_epoch, best_val, extra


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(epoch: int,
                    total_epochs: int,
                    model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: Optional[Any],
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    device: torch.device,
                    args,
                    logger: Logger) -> Dict[str, float]:
    model.train()

    meter_total = AverageMeter()
    meter_seg = AverageMeter()
    meter_sr = AverageMeter()
    meter_dice = AverageMeter()

    t0 = time.perf_counter()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs-1} [Train]", ncols=120, leave=False)

    for step, batch in enumerate(pbar):
        # Expect batch=(img, label)
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images = ensure_4d_image(images).float()
        labels_3d = ensure_label_shape(labels)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            # Warmup sr_only also needs SSRAE outputs
            need_aux = bool(args.use_ssrae)
            seg_logits, aux = forward_with_aux(model, images, labels_3d, need_aux=need_aux)
            loss, logs = criterion((seg_logits, aux), labels_3d, epoch=epoch, total_epochs=total_epochs)

        if args.amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # Scheduler stepping
        if scheduler is not None:
            if args.scheduler.lower() == "onecycle":
                scheduler.step()  # per-batch
            # cosine/step: per-epoch stepping handled outside

        # meters
        bs = images.size(0)
        meter_total.update(float(loss.detach().cpu()), bs)
        meter_seg.update(float(logs.get("seg/total", 0.0)), bs)
        meter_sr.update(float(logs.get("sr/total", 0.0)), bs)

        if args.report_dice:
            dice = mean_dice_from_logits(seg_logits.detach(), labels_3d.detach(),
                                         num_classes=args.num_classes,
                                         ignore_index=args.ignore_index,
                                         include_background=args.dice_include_bg)
            meter_dice.update(dice, bs)

        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({
            "lr": f"{lr_now:.2e}",
            "loss": f"{meter_total.avg:.4f}",
            "seg": f"{meter_seg.avg:.4f}",
            "sr": f"{meter_sr.avg:.4f}",
            "lam": f"{logs.get('lambda_sr', 0.0):.3f}",
            "dice": f"{meter_dice.avg:.4f}" if args.report_dice else "NA",
        })

    dt = time.perf_counter() - t0
    pbar.close()

    out = {
        "train/loss": meter_total.avg,
        "train/seg": meter_seg.avg,
        "train/sr": meter_sr.avg,
        "train/dice": meter_dice.avg if args.report_dice else float("nan"),
        "train/time_sec": dt,
    }
    return out


@torch.no_grad()
def validate_one_epoch(epoch: int,
                       total_epochs: int,
                       model: nn.Module,
                       loader: DataLoader,
                       criterion: nn.Module,
                       device: torch.device,
                       args,
                       logger: Logger) -> Dict[str, float]:
    model.eval()

    meter_total = AverageMeter()
    meter_seg = AverageMeter()
    meter_sr = AverageMeter()
    meter_dice = AverageMeter()

    t0 = time.perf_counter()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs-1} [Val  ]", ncols=120, leave=False)

    for batch in pbar:
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images = ensure_4d_image(images).float()
        labels_3d = ensure_label_shape(labels)

        need_aux = bool(args.use_ssrae)
        seg_logits, aux = forward_with_aux(model, images, labels_3d, need_aux=need_aux)
        loss, logs = criterion((seg_logits, aux), labels_3d, epoch=epoch, total_epochs=total_epochs)

        bs = images.size(0)
        meter_total.update(float(loss.detach().cpu()), bs)
        meter_seg.update(float(logs.get("seg/total", 0.0)), bs)
        meter_sr.update(float(logs.get("sr/total", 0.0)), bs)

        if args.report_dice:
            dice = mean_dice_from_logits(seg_logits.detach(), labels_3d.detach(),
                                         num_classes=args.num_classes,
                                         ignore_index=args.ignore_index,
                                         include_background=args.dice_include_bg)
            meter_dice.update(dice, bs)

        pbar.set_postfix({
            "loss": f"{meter_total.avg:.4f}",
            "seg": f"{meter_seg.avg:.4f}",
            "sr": f"{meter_sr.avg:.4f}",
            "dice": f"{meter_dice.avg:.4f}" if args.report_dice else "NA",
        })

    dt = time.perf_counter() - t0
    pbar.close()

    out = {
        "val/loss": meter_total.avg,
        "val/seg": meter_seg.avg,
        "val/sr": meter_sr.avg,
        "val/dice": meter_dice.avg if args.report_dice else float("nan"),
        "val/time_sec": dt,
    }
    return out


# -------------------------
# Main
# -------------------------
def build_scheduler(args, optimizer, steps_per_epoch: int):
    sch = args.scheduler.lower()
    if sch == "none":
        return None

    if sch == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if sch == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    if sch == "onecycle":
        # OneCycle needs total_steps
        total_steps = args.epochs * steps_per_epoch
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            pct_start=args.onecycle_pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,
        )

    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def parse_args():
    p = argparse.ArgumentParser("FaRMamba Trainer (single split, user-friendly)")

    # Experiment
    p.add_argument("--exp_name", type=str, default="farmamba_exp")
    p.add_argument("--save_root", type=str, default="./runs_farmamba")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=0.0)

    # Dataset
    p.add_argument("--dataset", type=str, default="paireddata", help="built-in: paireddata")
    p.add_argument("--dataset_root", type=str, default="")
    p.add_argument("--dataset_cls", type=str, default="", help='e.g. "mypkg.data:MyDataset"')
    p.add_argument("--dataset_kwargs", type=str, default="{}", help="JSON string of kwargs for dataset_cls")
    p.add_argument("--train_target", type=str, default="train")
    p.add_argument("--use_num", type=int, default=-1)
    p.add_argument("--val_ratio", type=float, default=0.2)

    # Model
    p.add_argument("--model", type=str, default="farmamba")
    p.add_argument("--model_cls", type=str, default="", help='e.g. "mypkg.models:Net"')
    p.add_argument("--model_kwargs", type=str, default="{}", help="JSON string of kwargs for model_cls")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--in_chans", type=int, default=1)
    p.add_argument("--num_classes", type=int, default=2)

    # Ablations (FaRMamba)
    p.add_argument("--use_msfm", action="store_true")
    p.add_argument("--msfm_type", type=str, default="WT", choices=["WT", "FFT", "DCT", "NONE"])
    p.add_argument("--use_ssrae", action="store_true")
    p.add_argument("--share_mamba_backbone", action="store_true")
    p.add_argument("--use_region_attention", action="store_true")
    p.add_argument("--region_attn_mode", type=str, default="hard", choices=["hard", "soft"])
    p.add_argument("--region_attn_beta", type=float, default=4.0)
    p.add_argument("--use_fusion", action="store_true")

    # Degrade params (SSRAE)
    p.add_argument("--degrade_scale_factor", type=float, default=0.5)
    p.add_argument("--degrade_blur_kernel", type=int, default=3)
    p.add_argument("--degrade_blur_sigma", type=float, default=1.0)
    p.add_argument("--degrade_noise_std", type=float, default=0.01)
    p.add_argument("--degrade_noise_train_only", action="store_true")
    p.add_argument("--degrade_blur_learnable", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])

    # Scheduler
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "step", "none"])
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--step_gamma", type=float, default=0.1)
    p.add_argument("--onecycle_pct_start", type=float, default=0.1)

    # Eval & saving
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--save_every", type=int, default=0, help="save epoch_{k}.pth every N epochs; 0=disable")
    p.add_argument("--report_dice", action="store_true")
    p.add_argument("--dice_include_bg", action="store_true")
    p.add_argument("--ignore_index", type=int, default=-1, help="-1 means no ignore_index")

    # Loss (FaRMambaLoss)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--lambda_mode", type=str, default="ema_ratio", choices=["ema_ratio", "fixed"])
    p.add_argument("--lambda_sr_fixed", type=float, default=0.1)
    p.add_argument("--lambda_sr_max", type=float, default=1.0)
    p.add_argument("--ema_momentum", type=float, default=0.9)
    # Warmup policy
    p.add_argument("--warmup_mode", type=str, default="sr_only",
                   choices=["sr_only", "seg_only", "joint"],
                   help="Warmup phase objective: sr_only=pretrain SSRAE, seg_only=seg only, joint=seg+sr from start")
    p.add_argument("--sr_warmup_weight", type=float, default=1.0,
                   help="Total loss weight for SR during warmup when warmup_mode=sr_only")

    # Recon schedule weights (linear)
    p.add_argument("--rec_l1_start", type=float, default=1.0)
    p.add_argument("--rec_l1_end", type=float, default=1.0)
    p.add_argument("--rec_cos_start", type=float, default=1.0)
    p.add_argument("--rec_cos_end", type=float, default=0.0)
    p.add_argument("--rec_grad_start", type=float, default=1.0)
    p.add_argument("--rec_grad_end", type=float, default=0.0)

    # Optional ROI-weighted reconstruction (strict paper-alignment: keep off)
    p.add_argument("--use_roi_weight", action="store_true")
    p.add_argument("--roi_boost", type=float, default=1.0)

    # Early stopping
    p.add_argument("--early_stop", action="store_true")
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    return p.parse_args()


def main():
    args = parse_args()

    # Post-process ignore_index
    if args.ignore_index < 0:
        args.ignore_index = None

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    run_name = f"{args.exp_name}_{now_str()}"
    save_dir = ensure_dir(os.path.join(args.save_root, run_name))
    ckpt_dir = ensure_dir(os.path.join(save_dir, "checkpoints"))
    fig_dir = ensure_dir(os.path.join(save_dir, "figures"))

    logger = Logger(os.path.join(save_dir, "train.log"))
    logger.log("=" * 80)
    logger.log(f"[RUN] {run_name}")
    logger.log(f"[DIR] {save_dir}")
    logger.log(f"[DEV] device={device}")
    logger.log("=" * 80)

    # Save config
    save_json(vars(args), os.path.join(save_dir, "config.json"))

    # Dataset
    if not args.dataset_root and not args.dataset_cls:
        logger.log("[WARN] dataset_root is empty. If using PairedData, please set --dataset_root.", also_print=True)

    full_ds = build_dataset(args, logger)
    n_total = len(full_ds)
    n_val = int(round(n_total * args.val_ratio))
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError(f"Bad split: total={n_total}, train={n_train}, val={n_val}. Adjust --val_ratio.")

    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    logger.log(f"[DATA] total={n_total}, train={len(train_ds)}, val={len(val_ds)} (val_ratio={args.val_ratio})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )

    # Model
    model = build_model(args, device, logger)

    # Loss
    try:
        from loss_farmamba import FaRMambaLoss, ReconSchedule
    except Exception as e:
        raise ImportError(
            "Failed to import FaRMambaLoss from loss_farmamba.py. "
            "Please make sure loss_farmamba.py is in the same folder or on PYTHONPATH."
        ) from e
    logger.log(
        f"[LOSS] warmup_epochs={args.warmup_epochs}, warmup_mode={args.warmup_mode}, sr_warmup_weight={args.sr_warmup_weight}")

    recon_schedule = ReconSchedule(
        l1_start=args.rec_l1_start, l1_end=args.rec_l1_end,
        cos_start=args.rec_cos_start, cos_end=args.rec_cos_end,
        grad_start=args.rec_grad_start, grad_end=args.rec_grad_end,
    )

    criterion = FaRMambaLoss(
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        use_ssrae=args.use_ssrae,
        warmup_epochs=args.warmup_epochs,

        # 新增：warmup 训 SSRAE 的策略
        warmup_mode=args.warmup_mode,
        sr_warmup_weight=args.sr_warmup_weight,

        recon_schedule=recon_schedule,
        lambda_mode=args.lambda_mode,
        lambda_sr_fixed=args.lambda_sr_fixed,
        lambda_sr_max=args.lambda_sr_max,
        ema_momentum=args.ema_momentum,
        use_roi_weight=args.use_roi_weight,
        roi_boost=args.roi_boost,
    ).to(device)

    # Optimizer
    opt_name = args.optimizer.lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Scheduler
    scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader))

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Resume
    start_epoch = 0
    best_val = float("inf")
    early_stop_bad = 0
    if args.resume:
        logger.log(f"[CKPT] Resuming from: {args.resume}")
        start_epoch, best_val, extra = load_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location="cpu"
        )
        early_stop_bad = int(extra.get("early_stop_bad", 0))
        logger.log(f"[CKPT] start_epoch={start_epoch}, best_val={best_val:.6f}, early_stop_bad={early_stop_bad}")

    # History
    history = {
        "train/loss": [],
        "train/seg": [],
        "train/sr": [],
        "train/dice": [],
        "val/loss": [],
        "val/seg": [],
        "val/sr": [],
        "val/dice": [],
        "lr": [],
        "epoch_time_sec": [],
        "train_time_sec": [],
        "val_time_sec": [],
    }

    metrics_csv = os.path.join(save_dir, "metrics.csv")
    times_csv = os.path.join(save_dir, "times.csv")

    logger.log("[START] Training begins.")
    t_global0 = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.perf_counter()

        # Train
        train_stats = train_one_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            logger=logger
        )

        # Per-epoch scheduler step (cosine/step)
        if scheduler is not None and args.scheduler.lower() in {"cosine", "step"}:
            scheduler.step()

        # Eval
        do_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs - 1)
        if do_eval:
            val_stats = validate_one_epoch(
                epoch=epoch,
                total_epochs=args.epochs,
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                args=args,
                logger=logger
            )
        else:
            val_stats = {
                "val/loss": float("nan"),
                "val/seg": float("nan"),
                "val/sr": float("nan"),
                "val/dice": float("nan"),
                "val/time_sec": 0.0,
            }

        epoch_dt = time.perf_counter() - epoch_t0
        lr_now = optimizer.param_groups[0]["lr"]

        # Log summary
        msg = (
            f"[E{epoch:03d}/{args.epochs-1}] "
            f"lr={lr_now:.2e} | "
            f"train: loss={train_stats['train/loss']:.4f} seg={train_stats['train/seg']:.4f} sr={train_stats['train/sr']:.4f}"
        )
        if args.report_dice:
            msg += f" dice={train_stats['train/dice']:.4f}"
        if do_eval:
            msg += f" | val: loss={val_stats['val/loss']:.4f} seg={val_stats['val/seg']:.4f} sr={val_stats['val/sr']:.4f}"
            if args.report_dice:
                msg += f" dice={val_stats['val/dice']:.4f}"
        msg += f" | time={human_time(epoch_dt)} (train={human_time(train_stats['train/time_sec'])}, val={human_time(val_stats['val/time_sec'])})"
        logger.log(msg)

        # Save history
        history["train/loss"].append(train_stats["train/loss"])
        history["train/seg"].append(train_stats["train/seg"])
        history["train/sr"].append(train_stats["train/sr"])
        history["train/dice"].append(train_stats["train/dice"])
        history["val/loss"].append(val_stats["val/loss"])
        history["val/seg"].append(val_stats["val/seg"])
        history["val/sr"].append(val_stats["val/sr"])
        history["val/dice"].append(val_stats["val/dice"])
        history["lr"].append(lr_now)
        history["epoch_time_sec"].append(epoch_dt)
        history["train_time_sec"].append(train_stats["train/time_sec"])
        history["val_time_sec"].append(val_stats["val/time_sec"])

        # CSV
        append_csv_row(
            metrics_csv,
            header=("epoch", "lr", "train_loss", "train_seg", "train_sr", "train_dice", "val_loss", "val_seg", "val_sr", "val_dice"),
            row=(epoch, lr_now,
                 train_stats["train/loss"], train_stats["train/seg"], train_stats["train/sr"], train_stats["train/dice"],
                 val_stats["val/loss"], val_stats["val/seg"], val_stats["val/sr"], val_stats["val/dice"])
        )
        append_csv_row(
            times_csv,
            header=("epoch", "epoch_time_sec", "train_time_sec", "val_time_sec", "total_elapsed_sec"),
            row=(epoch, epoch_dt, train_stats["train/time_sec"], val_stats["val/time_sec"], time.perf_counter() - t_global0)
        )

        # Checkpointing
        # Always save last
        extra = {"early_stop_bad": early_stop_bad}
        save_checkpoint(
            os.path.join(ckpt_dir, "last.pth"),
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, best_val=best_val, extra=extra
        )

        # Periodic save
        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"),
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, best_val=best_val, extra=extra
            )

        # Best save + early stopping
        if do_eval and not math.isnan(val_stats["val/loss"]):
            improved = (best_val - val_stats["val/loss"]) > args.early_stop_min_delta
            if improved:
                best_val = float(val_stats["val/loss"])
                early_stop_bad = 0
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pth"),
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=epoch, best_val=best_val, extra={"early_stop_bad": early_stop_bad}
                )
                logger.log(f"[BEST] Updated best val_loss={best_val:.6f} at epoch={epoch}. Saved to checkpoints/best.pth")
            else:
                early_stop_bad += 1

            if args.early_stop:
                if early_stop_bad >= args.early_stop_patience:
                    logger.log(
                        f"[EARLY-STOP] Triggered at epoch={epoch}: "
                        f"no improvement for {early_stop_bad} evals (patience={args.early_stop_patience})."
                    )
                    break

        # Plot (lightweight)
        if do_eval and (epoch % max(1, args.eval_interval) == 0):
            plot_curves(fig_dir, history)

    total_dt = time.perf_counter() - t_global0
    logger.log("=" * 80)
    logger.log(f"[DONE] Training finished. total_time={human_time(total_dt)} best_val={best_val:.6f}")
    logger.log(f"[ARTIFACTS] logs: {os.path.join(save_dir, 'train.log')}")
    logger.log(f"            metrics: {metrics_csv}")
    logger.log(f"            times: {times_csv}")
    logger.log(f"            checkpoints: {ckpt_dir}")
    logger.log("=" * 80)
    logger.close()


if __name__ == "__main__":
    main()
