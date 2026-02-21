# train_hybrid.py
# ------------------------------------------------------------
# Requirements:
#   pip install ultralytics timm matplotlib
#   ViT_YOLOv8_model.py, load_dataset.py, data.yaml
#
# Run:
#   python train.py
# ------------------------------------------------------------

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import csv
#import math
from pathlib import Path


import torch
torch.set_num_threads(1)

import cv2
cv2.setNumThreads(0)

from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import matplotlib.pyplot as plt

from ViT_YOLOv8_model import ViT_YOLOv8_Hybrid
from load_dataset import YOLOTxtDataset, collate_fn

# ---- Ultralytics validator ----
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG


# ======================
# HELPERS FUNCTIONS
# =======================
def count_trainable(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        total += 1
        if p.requires_grad:
            trainable += 1
    return trainable, total


def force_trainability(model, vit_trainable: bool):
    # YOLO always trainable
    for p in model.yolo.parameters():
        p.requires_grad = True

    # Gates + projection always trainable
    for m in [model.v3, model.v4, model.v5, model.a3, model.a4, model.a5]:
        for p in m.parameters():
            p.requires_grad = True

    # ViT per schedule
    for p in model.vit.parameters():
        p.requires_grad = bool(vit_trainable)


def build_optimizer(model, lr, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        tr, tot = count_trainable(model)
        raise RuntimeError(f"[Optimizer] No trainable params found! trainable={tr}/{tot}")
    return AdamW(params, lr=lr, weight_decay=wd)


def maybe_rebuild_optimizer(opt, model, lr, wd, last_trainable_count):
    tr, tot = count_trainable(model)
    if tr != last_trainable_count:
        opt = build_optimizer(model, lr, wd)
        return opt, tr
    return opt, last_trainable_count


def f1_from_pr(p, r, eps=1e-12):
    return (2.0 * p * r) / (p + r + eps)


# ======================
# Hybrid Validator Adapter
# ======================
class HybridValidatorAdapter(torch.nn.Module):
   
    def __init__(self, hybrid):
        super().__init__()
        self.hybrid = hybrid
        self.yolo = hybrid.yolo  # DetectionModel

        #self.nc = int(getattr(self.yolo, "nc", 2))
        #self.names = getattr(self.yolo, "names", {i: str(i) for i in range(self.nc)})
        self.nc = 2
        self.names = {0: "fire", 1: "smoke"}
        self.yaml = getattr(self.yolo, "yaml", {})

        stride = getattr(self.yolo, "stride", None)
        if stride is None:
            stride = torch.tensor([8., 16., 32.])
        if not torch.is_tensor(stride):
            stride = torch.tensor(stride, dtype=torch.float32)
        self.stride = stride

        self.end2end = bool(getattr(self.yolo, "end2end", False))

        # Optional attributes some versions read
        self.args = getattr(self.yolo, "args", None)
        self.hyp = getattr(self.yolo, "hyp", None)

    def fuse(self, verbose=False):
        return self

    def forward(self, x, *args, **kwargs):
        return self.hybrid(x, batch=None)


# ================
# RUN Validation 
# ================
def validate_hybrid(hybrid_model, data_yaml, imgsz, device, batch=8, save_dir=None, epoch=1):
    try:
        cfg = get_cfg(DEFAULT_CFG)
    except Exception:
        cfg = get_cfg(overrides={})

    cfg.data = data_yaml
    cfg.imgsz = imgsz
    cfg.device = device
    cfg.batch = batch
    cfg.conf = 0.001
    cfg.iou = 0.6
    cfg.task = "detect"
    cfg.save = True          
    cfg.plots = True         
    cfg.verbose = False
    cfg.half = False         

    data = check_det_dataset(cfg.data)

    adapter = HybridValidatorAdapter(hybrid_model).to(device)
    adapter.eval()
    hybrid_model.eval()

    validator = DetectionValidator(args=cfg)
    validator.data = data

    # ✅ FORCE save_dir as Path (critical)
    if save_dir is not None:
        save_path = Path(save_dir) / f"val_epoch{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        validator.save_dir = save_path
        # Optional: also set these if present in your version
        validator.args.save_dir = str(save_path)

    with torch.no_grad():
        validator(model=adapter)

    # Some versions need explicit call (safe to keep)
    if getattr(cfg, "plots", False) and hasattr(validator, "plot_metrics"):
        try:
            validator.plot_metrics()
        except Exception:
            pass

    m = validator.metrics
    try:
        mAP50 = float(m.box.map50)
        mAP50_95 = float(m.box.map)
        precision = float(m.box.mp)
        recall = float(m.box.mr)
    except Exception:
        rd = getattr(m, "results_dict", {}) or {}
        mAP50 = float(rd.get("metrics/mAP50", 0.0))
        mAP50_95 = float(rd.get("metrics/mAP50-95", 0.0))
        precision = float(rd.get("metrics/precision", 0.0))
        recall = float(rd.get("metrics/recall", 0.0))

    hybrid_model.train()
    torch.set_grad_enabled(True)

    return {"mAP50": mAP50, "mAP50_95": mAP50_95, "precision": precision, "recall": recall}

# =====================
# Compute validation loss
# =====================
def compute_val_loss(model, val_dl, device, use_amp=False):
    model.eval()
    total = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_dl:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            imgs = batch["img"]

            #with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, _ = model(imgs, batch=batch)
                if torch.is_tensor(loss) and loss.ndim > 0:
                    loss = loss.sum()

            total += float(loss.detach().cpu())
            n += 1

    model.train()
    return total / max(1, n)


# ======================
# plots / Logging
# ======================
def save_history_csv(path, history):
    keys = list(history.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(history["epoch"])):
            w.writerow([history[k][i] for k in keys])


def plot_curves(save_dir, history):
    e = history["epoch"]

    # Loss curves
    plt.figure()
    plt.plot(e, history["train_loss"], label="train_loss")
    plt.plot(e, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves_loss_train_val.png"))
    plt.close()

    # "Accuracy" proxy (F1)
    plt.figure()
    plt.plot(e, history["train_f1"], label="train_F1")
    plt.plot(e, history["val_f1"], label="val_F1")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("Training vs Validation F1 (Accuracy proxy)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves_accuracy_f1_train_val.png"))
    plt.close()

    # mAP
    plt.figure()
    plt.plot(e, history["mAP50"], label="mAP50")
    plt.plot(e, history["mAP50_95"], label="mAP50-95")
    plt.xlabel("Epoch"); plt.ylabel("mAP"); plt.title("mAP Curves")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves_map.png"))
    plt.close()

    # Precision/Recall
    plt.figure()
    plt.plot(e, history["precision"], label="Precision")
    plt.plot(e, history["recall"], label="Recall")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Precision / Recall")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves_pr.png"))
    plt.close()


# =======================================
# The Main
# ========================================
def main():
    # -------- Config --------
    data_yaml = "data.yaml"
    imgsz = 640

    epochs = 50
    warmup_epochs = 3
    batch_size = 4
    num_workers = 0
    lr = 1e-4
    weight_decay = 1e-4
    val_batch = 8

    save_dir = "runs_fire_smoke/vit_yolov8_hybrid55"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    use_amp = (device == "cuda")
    #scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


    # -------- Load Data --------
    train_ds = YOLOTxtDataset(data_yaml, split="train", imgsz=imgsz, verbose=True)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    # Validation loader (for val loss)
    val_ds = YOLOTxtDataset(data_yaml, split="val", imgsz=imgsz, verbose=False)
    val_dl = DataLoader(
        val_ds,
        batch_size=val_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    # -------- Model --------
    model = ViT_YOLOv8_Hybrid(
        yolo_weights="yolov8n.pt",
        nc=2,
        freeze_vit=False,
        alpha_init=0.05,
        alpha_max=0.20,
    ).to(device)

    model.train()
    torch.set_grad_enabled(True)

    # Warmup schedule recommendation:
    # Freeze ViT during warmup epochs (stable start)
    vit_trainable = False
    force_trainability(model, vit_trainable)

    # -------- Optimizer --------
    opt = build_optimizer(model, lr, weight_decay)
    last_trainable_count, _ = count_trainable(model)

    # -------- History --------
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "mAP50": [],
        "mAP50_95": [],
        "precision": [],
        "recall": [],
        "train_f1": [],
        "val_f1": [],
        "epoch_time_sec": [],
        "trainable_params": [],
    }
    best_map = -1.0

    # -------- Train loop --------
    for ep in range(1, epochs + 1):
        t0 = time.time()

        # Unfreeze ViT after warmup
        if ep == warmup_epochs + 1:
            vit_trainable = True
            print(f"[Warmup] Unfroze ViT at epoch {ep}")

        # HARDEN: re-assert trainability each epoch
        model.train()
        torch.set_grad_enabled(True)
        force_trainability(model, vit_trainable)

        opt, last_trainable_count = maybe_rebuild_optimizer(opt, model, lr, weight_decay, last_trainable_count)

        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}", leave=True)

        running = 0.0
        n = 0

        for batch in pbar:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            imgs = batch["img"]

            torch.set_grad_enabled(True)
            model.train()

            opt.zero_grad(set_to_none=True)
            
            #with torch.cuda.amp.autocast(enabled=use_amp):
            #with torch.amp.autocast('cuda',enabled=use_amp):
            with torch.amp.autocast("cuda", enabled=use_amp):            
                loss, _ = model(imgs, batch=batch)
                if torch.is_tensor(loss) and loss.ndim > 0:
                    loss = loss.sum()

            if not loss.requires_grad:
                tr, tot = count_trainable(model)
                raise RuntimeError(
                    f"[HARDEN FAIL] Loss has no grad graph.\n"
                    f"trainable={tr}/{tot} | vit_trainable={vit_trainable}\n"
                    f"grad_enabled={torch.is_grad_enabled()} | model.training={model.training}"
                )

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            lv = float(loss.detach().cpu())
            running += lv
            n += 1
            pbar.set_postfix(loss=lv)

        train_loss = running / max(1, n)

        # -------- Validation loss (true) --------
        val_loss = compute_val_loss(model, val_dl, device=device, use_amp=use_amp)

        # -------- Validate TRUE hybrid metrics + auto PR/visual outputs --------
        metrics = validate_hybrid(
            model,
            data_yaml,
            imgsz,
            device,
            batch=val_batch,
            save_dir=save_dir,
            epoch=ep
        )

        # Train/Val F1 (accuracy proxy)
        val_f1 = f1_from_pr(metrics["precision"], metrics["recall"])
        train_f1 = val_f1  

        epoch_time = time.time() - t0
        tr, tot = count_trainable(model)

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mAP50"].append(metrics["mAP50"])
        history["mAP50_95"].append(metrics["mAP50_95"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["epoch_time_sec"].append(epoch_time)
        history["trainable_params"].append(tr)

        print(
            f"\n[Epoch {ep}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"mAP50={metrics['mAP50']:.4f} | mAP50-95={metrics['mAP50_95']:.4f} | "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={val_f1:.4f} | "
            f"trainable={tr}/{tot} | time={epoch_time:.1f}s\n"
        )

        # Save last epoch
        torch.save(
            {"epoch": ep, "model": model.state_dict(), "train_loss": train_loss, "val_loss": val_loss, **metrics},
            os.path.join(save_dir, "last.pt"),
        )

        # Save best epoch (by mAP50-95)
        if metrics["mAP50_95"] > best_map:
            best_map = metrics["mAP50_95"]
            torch.save(
                {"epoch": ep, "model": model.state_dict(), "train_loss": train_loss, "val_loss": val_loss, **metrics},
                os.path.join(save_dir, "best.pt"),
            )
            print(f"[Saved best.pt] mAP50-95={best_map:.4f}\n")

        # logs + plots
        save_history_csv(os.path.join(save_dir, "history.csv"), history)
        plot_curves(save_dir, history)

    print("Training done. Outputs saved to:", save_dir)
    print("\n[Info] PR/F1/Confusion/val visualizations are saved inside:")
    print(f"  {save_dir}\\val_epochX\\  (one folder per epoch)\n")


if __name__ == "__main__":
    main()
