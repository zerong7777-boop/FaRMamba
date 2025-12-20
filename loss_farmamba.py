# -*- coding: utf-8 -*-
"""
loss_farmamba.py
Losses tailored for FaRMamba (paper-aligned):
  - Segmentation: Dice + CrossEntropy
  - SSRAE reconstruction: L1 + Cosine + Gradient(Sobel)
  - Optional: epoch-based linear scheduling + EMA-smoothed lambda_sr
Compatible with FaRMamba outputs:
  - seg_logits
  - or (seg_logits, aux_dict) where aux_dict contains:
      sr_pred, sr_target, region_mask (optional)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def _ensure_4d(x: Tensor) -> Tensor:
    if x.dim() == 4:
        return x
    if x.dim() == 3:  # [B,H,W] -> [B,1,H,W]
        return x.unsqueeze(1)
    raise ValueError(f"Expected 3D/4D tensor, got shape={tuple(x.shape)}")


def _labels_to_long(labels: Tensor, num_classes: int, ignore_index: Optional[int] = None) -> Tensor:
    """
    Accept labels as:
      - [B,H,W] int/float
      - [B,1,H,W]
      - [B,C,H,W] one-hot/prob
    Return: [B,H,W] long
    """
    if labels.dim() == 4:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            labels = labels.argmax(dim=1)
    elif labels.dim() == 3:
        pass
    else:
        raise ValueError(f"Invalid labels shape={tuple(labels.shape)}")

    if labels.dtype != torch.long:
        labels = labels.long()

    # Optional sanity clamp (except ignore_index)
    if ignore_index is not None:
        valid = labels != ignore_index
        if valid.any():
            labels_valid = labels[valid]
            labels_valid = labels_valid.clamp(min=0, max=num_classes - 1)
            labels = labels.clone()
            labels[valid] = labels_valid
    else:
        labels = labels.clamp(min=0, max=num_classes - 1)

    return labels


def _one_hot(labels_long: Tensor, num_classes: int, ignore_index: Optional[int] = None) -> Tensor:
    """
    labels_long: [B,H,W] long
    returns: [B,C,H,W] float in {0,1}; ignored pixels are all-zeros.
    """
    if labels_long.dim() != 3:
        raise ValueError(f"labels_long must be [B,H,W], got {tuple(labels_long.shape)}")

    B, H, W = labels_long.shape
    oh = torch.zeros((B, num_classes, H, W), device=labels_long.device, dtype=torch.float32)

    if ignore_index is None:
        oh.scatter_(1, labels_long.unsqueeze(1), 1.0)
        return oh

    valid = labels_long != ignore_index
    if valid.any():
        idx = labels_long[valid].clamp(min=0, max=num_classes - 1)
        b_idx, h_idx, w_idx = valid.nonzero(as_tuple=True)
        oh[b_idx, idx, h_idx, w_idx] = 1.0
    return oh


def soft_dice_loss(
    logits: Tensor,
    labels_long: Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
    smooth: float = 1.0,
    eps: float = 1e-7,
    include_background: bool = True,
) -> Tensor:
    """
    Multi-class soft Dice loss on logits (expects [B,C,H,W]).
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be [B,C,H,W], got {tuple(logits.shape)}")

    probs = F.softmax(logits, dim=1)
    target = _one_hot(labels_long, num_classes=num_classes, ignore_index=ignore_index).to(probs.dtype)

    if ignore_index is not None:
        valid = (labels_long != ignore_index).unsqueeze(1)  # [B,1,H,W]
        probs = probs * valid
        target = target * valid

    # Optionally exclude background channel 0
    if not include_background and num_classes > 1:
        probs = probs[:, 1:, ...]
        target = target[:, 1:, ...]
        denom_classes = num_classes - 1
    else:
        denom_classes = num_classes

    # Dice per class
    dims = (0, 2, 3)
    intersection = torch.sum(probs * target, dim=dims)
    probs_sum = torch.sum(probs, dim=dims)
    target_sum = torch.sum(target, dim=dims)

    dice = (2.0 * intersection + smooth) / (probs_sum + target_sum + smooth + eps)
    loss = 1.0 - dice  # [C] or [C-1]
    return loss.sum() / float(denom_classes)


def cross_entropy_loss(
    logits: Tensor,
    labels_long: Tensor,
    ignore_index: Optional[int] = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    return F.cross_entropy(
        logits,
        labels_long,
        ignore_index=-100 if ignore_index is None else int(ignore_index),
        label_smoothing=float(label_smoothing),
    )


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], device=device, dtype=dtype) / 4.0
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], device=device, dtype=dtype) / 4.0
    return kx, ky


def gradient_loss_sobel(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """
    Gradient loss based on Sobel filters.
    pred/target: [B,C,H,W]
    """
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    B, C, H, W = pred.shape
    kx, ky = _sobel_kernels(pred.device, pred.dtype)
    kx = kx.view(1, 1, 3, 3).repeat(C, 1, 1, 1)  # depthwise
    ky = ky.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    pred_gx = F.conv2d(pred, kx, padding=1, groups=C)
    pred_gy = F.conv2d(pred, ky, padding=1, groups=C)
    tar_gx = F.conv2d(target, kx, padding=1, groups=C)
    tar_gy = F.conv2d(target, ky, padding=1, groups=C)

    diff = (pred_gx - tar_gx).abs() + (pred_gy - tar_gy).abs()

    if reduction == "mean":
        return diff.mean()
    if reduction == "sum":
        return diff.sum()
    if reduction == "none":
        return diff
    raise ValueError(f"Unknown reduction: {reduction}")


def cosine_feature_loss(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Cosine loss on per-pixel feature vectors across channels.
    pred/target: [B,C,H,W]
    returns: mean(1 - cosine_similarity)
    """
    pred = _ensure_4d(pred)
    target = _ensure_4d(target)
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

    # Flatten spatial, compare channel vectors at each location
    B, C, H, W = pred.shape
    p = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
    t = target.permute(0, 2, 3, 1).contiguous().view(-1, C)

    p = p / (p.norm(dim=1, keepdim=True) + eps)
    t = t / (t.norm(dim=1, keepdim=True) + eps)
    cos = (p * t).sum(dim=1)  # [-1,1]
    loss = 1.0 - cos
    return loss.mean()


@dataclass
class ReconSchedule:
    """
    Linear schedule for reconstruction term weights (L1 / Cosine / Grad).
    w = w_start + (w_end - w_start) * (epoch/(total_epochs-1))
    """
    l1_start: float = 1.0
    l1_end: float = 1.0
    cos_start: float = 1.0
    cos_end: float = 0.0
    grad_start: float = 1.0
    grad_end: float = 0.0

    def weights(self, epoch: int, total_epochs: int) -> Tuple[float, float, float]:
        if total_epochs <= 1:
            return float(self.l1_end), float(self.cos_end), float(self.grad_end)
        t = float(max(0, min(epoch, total_epochs - 1))) / float(total_epochs - 1)
        w_l1 = self.l1_start + (self.l1_end - self.l1_start) * t
        w_cos = self.cos_start + (self.cos_end - self.cos_start) * t
        w_grad = self.grad_start + (self.grad_end - self.grad_start) * t
        return float(w_l1), float(w_cos), float(w_grad)


class FaRMambaLoss(nn.Module):
    """
    Paper-aligned loss for FaRMamba.

    Usage (typical):
      loss_fn = FaRMambaLoss(num_classes=K, warmup_epochs=10, lambda_mode="ema_ratio")
      total, logs = loss_fn(model_out, labels, epoch=e, total_epochs=E)

    model_out can be:
      - seg_logits (Tensor)
      - (seg_logits, aux_dict)
    aux_dict expected keys (if use_ssrae=True):
      - sr_pred: Tensor [B,48,H/2,W/2]
      - sr_target: Tensor [B,48,H/2,W/2]
      - region_mask: optional Tensor [B,1,h,w] (any res; will be resized)
    """

    def __init__(
        self,
        num_classes: int,
        # seg loss
        seg_dice_weight: float = 1.0,
        seg_ce_weight: float = 1.0,
        include_background: bool = True,
        ignore_index: Optional[int] = None,
        label_smoothing: float = 0.0,
        # ssrae loss
        use_ssrae: bool = True,
        warmup_epochs: int = 0,
        recon_schedule: Optional[ReconSchedule] = None,
        # lambda_sr strategy
        lambda_mode: str = "ema_ratio",  # "ema_ratio" or "fixed"
        lambda_sr_fixed: float = 0.1,
        lambda_sr_max: float = 1.0,
        ema_momentum: float = 0.9,
        # ROI weighting
        use_roi_weight: bool = False,
        roi_boost: float = 1.0,  # extra weight applied to ROI pixels: weight = 1 + roi_boost*mask
        warmup_mode: str = "sr_only",  # "seg_only" / "sr_only" / "joint"
        sr_warmup_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.seg_dice_weight = float(seg_dice_weight)
        self.seg_ce_weight = float(seg_ce_weight)
        self.include_background = bool(include_background)
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)

        self.use_ssrae = bool(use_ssrae)
        self.warmup_epochs = int(warmup_epochs)
        assert warmup_mode in {"sr_only", "seg_only", "joint"}, f"Invalid warmup_mode={warmup_mode}"
        self.warmup_mode = str(warmup_mode)
        self.sr_warmup_weight = float(sr_warmup_weight)

        self.recon_schedule = recon_schedule or ReconSchedule()

        assert lambda_mode in {"ema_ratio", "fixed"}, f"lambda_mode must be ema_ratio/fixed, got {lambda_mode}"
        self.lambda_mode = lambda_mode
        self.lambda_sr_fixed = float(lambda_sr_fixed)
        self.lambda_sr_max = float(lambda_sr_max)
        self.ema_momentum = float(ema_momentum)

        self.use_roi_weight = bool(use_roi_weight)
        self.roi_boost = float(roi_boost)

        # EMA state (registered buffer for checkpointing)
        self.register_buffer("_lambda_sr_ema", torch.tensor(0.0, dtype=torch.float32), persistent=True)
        self.register_buffer("_ema_inited", torch.tensor(0, dtype=torch.int32), persistent=True)



    def _compute_seg_losses(self, seg_logits: Tensor, labels: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        labels_long = _labels_to_long(labels, num_classes=self.num_classes, ignore_index=self.ignore_index)

        loss_dice = soft_dice_loss(
            seg_logits,
            labels_long,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
        )

        loss_ce = cross_entropy_loss(
            seg_logits,
            labels_long,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        seg = self.seg_dice_weight * loss_dice + self.seg_ce_weight * loss_ce
        logs = {
            "seg/dice": float(loss_dice.detach().cpu()),
            "seg/ce": float(loss_ce.detach().cpu()),
            "seg/total": float(seg.detach().cpu()),
        }
        return seg, logs

    def _roi_weight_map(self, roi_mask: Tensor, target_hw: Tuple[int, int], device, dtype) -> Tensor:
        """
        roi_mask: [B,1,h,w] or [B,h,w]
        returns: [B,1,H,W] weight map
        """
        roi_mask = _ensure_4d(roi_mask)
        if roi_mask.shape[1] != 1:
            roi_mask = roi_mask[:, :1]
        if roi_mask.shape[-2:] != target_hw:
            roi_mask = F.interpolate(roi_mask.float(), size=target_hw, mode="nearest")
        roi_mask = roi_mask.to(device=device, dtype=dtype)
        w = 1.0 + self.roi_boost * roi_mask
        return w

    def _compute_recon_losses(
        self,
        sr_pred: Tensor,
        sr_target: Tensor,
        epoch: int,
        total_epochs: int,
        roi_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        sr_pred = _ensure_4d(sr_pred)
        sr_target = _ensure_4d(sr_target)

        if sr_pred.shape != sr_target.shape:
            raise ValueError(f"sr_pred/sr_target shape mismatch: {tuple(sr_pred.shape)} vs {tuple(sr_target.shape)}")

        # optional ROI weighting
        weight_map = None
        if self.use_roi_weight and (roi_mask is not None):
            weight_map = self._roi_weight_map(roi_mask, sr_pred.shape[-2:], sr_pred.device, sr_pred.dtype)

        w_l1, w_cos, w_grad = self.recon_schedule.weights(epoch, total_epochs)

        # L1
        if weight_map is None:
            loss_l1 = F.l1_loss(sr_pred, sr_target)
        else:
            loss_l1 = (weight_map * (sr_pred - sr_target).abs()).mean()

        # Cosine
        loss_cos = cosine_feature_loss(sr_pred, sr_target)

        # Gradient
        loss_grad = gradient_loss_sobel(sr_pred, sr_target, reduction="mean")

        recon = w_l1 * loss_l1 + w_cos * loss_cos + w_grad * loss_grad
        logs = {
            "sr/l1": float(loss_l1.detach().cpu()),
            "sr/cos": float(loss_cos.detach().cpu()),
            "sr/grad": float(loss_grad.detach().cpu()),
            "sr/w_l1": float(w_l1),
            "sr/w_cos": float(w_cos),
            "sr/w_grad": float(w_grad),
            "sr/total": float(recon.detach().cpu()),
        }
        return recon, logs

    @torch.no_grad()
    def _update_lambda_sr_ema(self, seg_loss: Tensor, sr_loss: Tensor) -> float:
        # ratio = sr / (seg + sr)
        denom = (seg_loss.detach() + sr_loss.detach()).clamp_min(1e-8)
        ratio = (sr_loss.detach() / denom).clamp(0.0, 1.0).float()

        if int(self._ema_inited.item()) == 0:
            self._lambda_sr_ema.copy_(ratio)
            self._ema_inited.fill_(1)
        else:
            m = self.ema_momentum
            self._lambda_sr_ema.mul_(m).add_((1.0 - m) * ratio)

        lam = float(self._lambda_sr_ema.item()) * self.lambda_sr_max
        return max(0.0, min(lam, self.lambda_sr_max))

    def forward(
        self,
        model_out: Union[Tensor, Tuple[Tensor, Dict[str, Any]]],
        labels: Tensor,
        epoch: int = 0,
        total_epochs: int = 1,
    ) -> Tuple[Tensor, Dict[str, float]]:
        # unpack model output
        if isinstance(model_out, (tuple, list)) and len(model_out) == 2 and isinstance(model_out[1], dict):
            seg_logits, aux = model_out
        else:
            seg_logits, aux = model_out, {}

        if seg_logits.dim() != 4:
            raise ValueError(f"seg_logits must be [B,C,H,W], got {tuple(seg_logits.shape)}")

        # seg always computed for logging (cheap)
        seg_loss, seg_logs = self._compute_seg_losses(seg_logits, labels)

        sr_loss = seg_logits.new_tensor(0.0)
        sr_logs = {"sr/total": 0.0}
        lambda_sr = 0.0

        if self.use_ssrae:
            sr_pred = aux.get("sr_pred", None)
            sr_target = aux.get("sr_target", None)
            if (sr_pred is not None) and (sr_target is not None):
                roi_mask = aux.get("region_mask", None)
                sr_loss, sr_logs = self._compute_recon_losses(
                    sr_pred=sr_pred,
                    sr_target=sr_target,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    roi_mask=roi_mask,
                )

        # ---- WARMUP POLICY ----
        if self.use_ssrae and (epoch < self.warmup_epochs):
            if self.warmup_mode == "sr_only":
                total = float(self.sr_warmup_weight) * sr_loss
                logs = {}
                logs.update(seg_logs)
                logs.update(sr_logs)
                logs["lambda_sr"] = 0.0
                logs["loss/total"] = float(total.detach().cpu())
                logs["warmup_mode"] = 1.0
                return total, logs

            elif self.warmup_mode == "joint":
                # joint from the beginning (if you want)
                if self.lambda_mode == "fixed":
                    lambda_sr = float(self.lambda_sr_fixed)
                else:
                    lambda_sr = self._update_lambda_sr_ema(seg_loss, sr_loss)
                total = seg_loss + lambda_sr * sr_loss

            else:
                # seg_only
                total = seg_loss

        else:
            # ---- JOINT AFTER WARMUP ----
            if self.use_ssrae and (sr_loss is not None):
                if self.lambda_mode == "fixed":
                    lambda_sr = float(self.lambda_sr_fixed)
                else:
                    lambda_sr = self._update_lambda_sr_ema(seg_loss, sr_loss)
            total = seg_loss + lambda_sr * sr_loss

        logs = {}
        logs.update(seg_logs)
        logs.update(sr_logs)
        logs["lambda_sr"] = float(lambda_sr)
        logs["loss/total"] = float(total.detach().cpu())
        logs["warmup_mode"] = 0.0
        return total, logs


# ------------------------------
# Minimal self-test
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    B, K, H, W = 2, 4, 250, 330
    seg_logits = torch.randn(B, K, H, W, device=device)
    labels = torch.randint(0, K, (B, H, W), device=device)

    # Fake SSRAE
    sr_pred = torch.randn(B, 48, H // 2, W // 2, device=device)
    sr_target = torch.randn(B, 48, H // 2, W // 2, device=device)
    region_mask = torch.randint(0, 2, (B, 1, 8, 8), device=device).float()

    loss_fn = FaRMambaLoss(
        num_classes=K,
        warmup_epochs=0,
        lambda_mode="ema_ratio",
        use_ssrae=True,
        use_roi_weight=True,
        roi_boost=1.0,
        recon_schedule=ReconSchedule(
            l1_start=1.0, l1_end=1.0,
            cos_start=1.0, cos_end=0.0,
            grad_start=1.0, grad_end=0.0,
        )
    ).to(device)

    total, logs = loss_fn((seg_logits, {"sr_pred": sr_pred, "sr_target": sr_target, "region_mask": region_mask}),
                          labels, epoch=5, total_epochs=100)
    print("[OK] total =", float(total.detach().cpu()))
    for k in sorted(logs.keys()):
        print(f"  {k}: {logs[k]}")
