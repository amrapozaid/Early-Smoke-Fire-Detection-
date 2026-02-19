import os
import glob
import yaml
import cv2
import torch
from torch.utils.data import Dataset


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir, p):
    p = str(p).strip()
    p = p.replace("/", os.sep)
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(base_dir, p))


def find_images(img_dir):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
    return sorted(set(files))


def img2label_path(img_path):
    # Ultralytics convention
    p = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    return os.path.splitext(p)[0] + ".txt"


class YOLOTxtDataset(Dataset):
    """
    This class updated for support multiple YOLO based datasets from data.yaml.

    Supports:
      - Detection labels: cls x y w h
      - Detection+conf:   cls x y w h conf    (conf ignored)
      - Segmentation:     cls x1 y1 x2 y2 ... (converted to bbox)    
    """

    def __init__(self, data_yaml, split="train", imgsz=640, verbose=True):
        self.data_yaml = os.path.abspath(data_yaml)
        self.yaml_dir = os.path.dirname(self.data_yaml)
        self.imgsz = int(imgsz)
        self.verbose = bool(verbose)

        d = load_data_yaml(self.data_yaml)

        # Handle optional base path
        base_from_yaml = d.get("path", None)
        base_dir = resolve_path(self.yaml_dir, base_from_yaml) if base_from_yaml else self.yaml_dir

        if split not in d:
            raise KeyError(f"Split '{split}' not found in data.yaml keys: {list(d.keys())}")

        self.img_dir = resolve_path(base_dir, d[split])

        if self.verbose:
            print(f"[Dataset] data.yaml: {self.data_yaml}")
            print(f"[Dataset] base_dir:  {base_dir}")
            print(f"[Dataset] img_dir:   {self.img_dir}")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory does not exist: {self.img_dir}")

        self.img_files = find_images(self.img_dir)

        if self.verbose:
            print(f"[Dataset] Found {len(self.img_files)} images.")

        if len(self.img_files) == 0:
            parent = os.path.dirname(self.img_dir)
            hint = ""
            if os.path.isdir(parent):
                hint = "\nFolders under parent:\n" + "\n".join([" - " + x for x in sorted(os.listdir(parent))[:50]])
            raise FileNotFoundError(f"No images found in {self.img_dir}{hint}")

        self.label_files = [img2label_path(p) for p in self.img_files]
        self._warned_label_files = set()

    @staticmethod
    def _poly_to_bbox(coords):
        """coords: [x1,y1,x2,y2,...] normalized."""
        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x = (x_min + x_max) / 2.0
        y = (y_min + y_max) / 2.0
        w = (x_max - x_min)
        h = (y_max - y_min)
        return x, y, w, h

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        img_path = self.img_files[i]
        im = cv2.imread(img_path)
        if im is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        h0, w0 = im.shape[:2]

        im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0

        labels = []
        lp = self.label_files[i]
        if os.path.exists(lp):
            with open(lp, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        # bad line
                        if self.verbose and lp not in self._warned_label_files:
                            print(f"[WARN] Bad label line (<5 tokens) in {lp}: '{line[:80]}'")
                            self._warned_label_files.add(lp)
                        continue

                    # 1) cls x y w h
                    if len(parts) == 5:
                        cls, x, y, w, h = parts

                    # 2) cls x y w h conf  (ignore conf)
                    elif len(parts) == 6:
                        cls, x, y, w, h, _conf = parts

                    # 3) segmentation polygon: cls x1 y1 x2 y2 ...
                    else:
                        cls = parts[0]
                        try:
                            coords = list(map(float, parts[1:]))
                        except ValueError:
                            if self.verbose and lp not in self._warned_label_files:
                                print(f"[WARN] Non-numeric polygon in {lp}: '{line[:80]}'")
                                self._warned_label_files.add(lp)
                            continue

                        box = self._poly_to_bbox(coords)
                        if box is None:
                            continue
                        x, y, w, h = box

                    try:
                        cls_i = int(float(cls))
                        x = float(x); y = float(y); w = float(w); h = float(h)
                    except ValueError:
                        if self.verbose and lp not in self._warned_label_files:
                            print(f"[WARN] Non-numeric label in {lp}: '{line[:80]}'")
                            self._warned_label_files.add(lp)
                        continue

                    # sanity clamp to [0,1] where applicable (robustness)
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))

                    labels.append([cls_i, x, y, w, h])

        labels = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)

        return {"img": im, "labels": labels, "im_file": img_path, "ori_shape": (h0, w0)}


def collate_fn(batch):
    imgs = torch.stack([b["img"] for b in batch], 0)

    # targets: [image_index, class, x, y, w, h]
    targets = []
    for i, b in enumerate(batch):
        if b["labels"].numel():
            t = torch.zeros((b["labels"].shape[0], 6), dtype=torch.float32)
            t[:, 0] = i
            t[:, 1:] = b["labels"]
            targets.append(t)

    if len(targets):
        targets = torch.cat(targets, 0)
        cls = targets[:, 1:2]
        bboxes = targets[:, 2:6]
        batch_idx = targets[:, 0]
    else:
        cls = torch.zeros((0, 1), dtype=torch.float32)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.float32)

    return {
        "img": imgs,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": [b["im_file"] for b in batch],
    }
