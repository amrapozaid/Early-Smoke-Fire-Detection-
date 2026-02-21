# hybrid_model.py
# An Intelligent Approach for Early Smoke/Fire Detection Using Vision Sensors in Smart Cities
# Vision Transformer (ViT) + YOLOv8 
# ------------------------------------------------------------

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from ultralytics import YOLO


# ---------------------------
# Utilities
# ---------------------------
def _to_namespace(obj: Any) -> SimpleNamespace:
    if obj is None:
        return SimpleNamespace()
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**obj)
    ns = SimpleNamespace()
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            setattr(ns, k, getattr(obj, k))
        except Exception:
            pass
    return ns


def _force_defaults(ns: SimpleNamespace, defaults: dict) -> SimpleNamespace:
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def ensure_ultra_hyp(detection_model):
    defaults = dict(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        fl_gamma=0.0,
        iou=0.7,
        anchor_t=4.0,
    )

    args_ns = _force_defaults(_to_namespace(getattr(detection_model, "args", None)), defaults)
    hyp_ns = _force_defaults(_to_namespace(getattr(detection_model, "hyp", None)), defaults)

    detection_model.args = args_ns
    detection_model.hyp = hyp_ns
    return detection_model


class ResidualGate(nn.Module):
    def __init__(self, init: float = 0.05, max_scale: float = 0.20):
        super().__init__()
        init = float(init)
        max_scale = float(max_scale)

        init = max(1e-4, min(max_scale - 1e-4, init))
        p = init / max_scale
        self.logit = nn.Parameter(torch.logit(torch.tensor(p)))
        self.max_scale = max_scale

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit) * self.max_scale


class ViT_YOLOv8_Hybrid(nn.Module):

    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        nc: int = 2,
        vit_name: str = "vit_base_patch16_224",
        vit_pretrained: bool = True,
        vit_imgsz: int = 224,
        freeze_vit: bool = True,
        alpha_init: float = 0.05,
        alpha_max: float = 0.20,
    ):
        super().__init__()

        # ---- Load YOLO8 ----
        y = YOLO(yolo_weights)
        self.yolo = y.model
        
        self.yolo.nc = int(nc)
        ensure_ultra_hyp(self.yolo)

        # ---- ViT ----
        self.vit_imgsz = int(vit_imgsz)
        self.vit = timm.create_model(vit_name, pretrained=vit_pretrained, num_classes=0)
        self.vit_dim = getattr(self.vit, "num_features", None) or getattr(self.vit, "embed_dim")

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # ---- Detect head ----
        self.detect = self.yolo.model[-1]

        # ---- Infer P3/P4/P5 channels ----
        ch_p3, ch_p4, ch_p5 = self._infer_detect_in_channels()

        # Projections 
        self.v3 = nn.Sequential(nn.Conv2d(self.vit_dim, ch_p3, 1, bias=False), nn.BatchNorm2d(ch_p3), nn.SiLU())
        self.v4 = nn.Sequential(nn.Conv2d(self.vit_dim, ch_p4, 1, bias=False), nn.BatchNorm2d(ch_p4), nn.SiLU())
        self.v5 = nn.Sequential(nn.Conv2d(self.vit_dim, ch_p5, 1, bias=False), nn.BatchNorm2d(ch_p5), nn.SiLU())

        # Gates 
        self.a3 = ResidualGate(alpha_init, alpha_max)
        self.a4 = ResidualGate(alpha_init, alpha_max)
        self.a5 = ResidualGate(alpha_init, alpha_max)

    def _infer_detect_in_channels(self) -> Tuple[int, int, int]:
        if hasattr(self.detect, "ch") and isinstance(self.detect.ch, (list, tuple)) and len(self.detect.ch) == 3:
            return int(self.detect.ch[0]), int(self.detect.ch[1]), int(self.detect.ch[2])

        device = next(self.yolo.parameters()).device
        x = torch.zeros(1, 3, 640, 640, device=device)
        feats = self._forward_yolo_to_feats(x)
        return int(feats[0].shape[1]), int(feats[1].shape[1]), int(feats[2].shape[1])

    def _forward_yolo_to_feats(self, imgs: torch.Tensor) -> List[torch.Tensor]:
        x = imgs
        y: List[torch.Tensor] = []

        for m in self.yolo.model[:-1]:
            f = getattr(m, "f", -1)
            if f != -1:
                if isinstance(f, int):
                    x_in = y[f] if f != -1 else x
                else:
                    x_in = [x if j == -1 else y[j] for j in f]
            else:
                x_in = x

            x = m(x_in)
            y.append(x)

        f_det = getattr(self.detect, "f", None)
        if not isinstance(f_det, (list, tuple)) or len(f_det) != 3:
            return [y[-3], y[-2], y[-1]]
        return [y[i] for i in f_det]

    @torch.no_grad()
    def _vit_grid_hw(self) -> Tuple[int, int]:
        patch = getattr(self.vit, "patch_embed", None)
        if patch is not None and hasattr(patch, "patch_size"):
            ps = patch.patch_size
            ps = ps[0] if isinstance(ps, tuple) else int(ps)
            g = self.vit_imgsz // ps
            return g, g
        return 14, 14

    def vit_to_map(self, imgs: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(imgs, size=(self.vit_imgsz, self.vit_imgsz), mode="bilinear", align_corners=False)
        feats = self.vit.forward_features(x)

        if feats.dim() == 4:
            return feats

        B, N, C = feats.shape
        gh, gw = self._vit_grid_hw()

        if N == 1 + gh * gw:
            feats = feats[:, 1:, :]
            N = feats.shape[1]

        if N != gh * gw:
            g = int(math.sqrt(N))
            gh, gw = g, g

        return feats.transpose(1, 2).contiguous().view(B, C, gh, gw)

    def forward(self, imgs: torch.Tensor, batch: Optional[dict] = None):
       
        if batch is not None:
            self.yolo.train()
            self.detect.train()

        p3, p4, p5 = self._forward_yolo_to_feats(imgs)

        # ---- ViT -> map ----
        vit_map = self.vit_to_map(imgs)
        v3 = self.v3(F.interpolate(vit_map, size=p3.shape[-2:], mode="bilinear", align_corners=False))
        v4 = self.v4(F.interpolate(vit_map, size=p4.shape[-2:], mode="bilinear", align_corners=False))
        v5 = self.v5(F.interpolate(vit_map, size=p5.shape[-2:], mode="bilinear", align_corners=False))

        # ---- Residual fusion ----
        p3f = p3 + self.a3() * v3
        p4f = p4 + self.a4() * v4
        p5f = p5 + self.a5() * v5

        # ---- Detect on fused ----
        preds = self.detect([p3f, p4f, p5f])

        if batch is None:
            return preds

        # ---- Loss ----
        ensure_ultra_hyp(self.yolo)
        try:
            loss, items = self.yolo.loss(batch, preds)
        except TypeError:
            # older ultralytics signatures
            loss, items = self.yolo.loss(batch)

        return loss, items
