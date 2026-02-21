# predict.py
# ------------------------------------------------------------
# Inference for ViT_YOLOv8_Hybrid (raw tensor output + NMS)
# ------------------------------------------------------------

import os
import cv2
import torch
#import numpy as np

try:
    from ultralytics.utils.nms import non_max_suppression
except Exception:
    from ultralytics.utils.ops import non_max_suppression

from ViT_YOLOv8_model import ViT_YOLOv8_Hybrid  # adjust if filename differs


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """YOLO-style letterbox resize. Returns resized image, scale, padw, padh."""
    h, w = im.shape[:2]
    scale = new_shape / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h = new_shape - nh
    pad_w = new_shape - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, scale, left, top


def scale_coords_xyxy(boxes, scale, padw, padh, orig_w, orig_h):
    """Map xyxy boxes from letterboxed image back to original image coords."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= padw
    boxes[:, [1, 3]] -= padh
    boxes[:, :4] /= scale

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
    return boxes


def draw_boxes(img, boxes, scores, classes, names):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        cls = int(cls)

        label = f"{names.get(cls, str(cls))} {float(score):.2f}"
        color = (0, 0, 255) if names.get(cls, "").lower() == "fire" else (255, 0, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def main():
    # ------------------------
    # CONFIG 
    # ------------------------
    weights_path = "val/best.pt"
    image_path = "val/test_image.jpg"
    imgsz = 640
    conf = 0.25
    iou = 0.6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ------------------------
    # Load Model (MUST match training)
    # ------------------------
    model = ViT_YOLOv8_Hybrid(
        yolo_weights="yolov8n.pt",
        nc=2,
        freeze_vit=False
    ).to(device)

    
    model.yolo.nc = 2
    model.yolo.names = {0: "fire", 1: "smoke"}

    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Model loaded]")
    print("Device:", device)
    print("Missing keys:", len(missing), "| Unexpected keys:", len(unexpected))
    print("YOLO nc:", getattr(model.yolo, "nc", None))
    print("YOLO names:", getattr(model.yolo, "names", None))

    model.eval()

    # ------------------------
    # Load Image
    # ------------------------
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise ValueError(f"Image not found: {image_path}")

    h0, w0 = img0.shape[:2]

    img_lb, scale, padw, padh = letterbox(img0, imgsz)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # ------------------------
    # Inference (raw preds)
    # ------------------------
    with torch.no_grad():
        preds = model(img_tensor)

    # ------------------------
    # NMS
    # ------------------------
    preds = non_max_suppression(preds, conf_thres=conf, iou_thres=iou, max_det=300)
    det = preds[0]

    if det is None or len(det) == 0:
        print("No detections.")
        return

    det = det.detach().cpu().numpy()
    boxes = det[:, :4]
    scores = det[:, 4]
    classes = det[:, 5].astype(int)

    # ------------------------
    # Rescale boxes to original image
    # ------------------------
    boxes = scale_coords_xyxy(boxes, scale, padw, padh, w0, h0)
    #print("boxes:", boxes)

    # ------------------------
    # Names
    # ------------------------
    names = {0: "fire", 1: "smoke"}
    if hasattr(model.yolo, "names") and model.yolo.names:
        if isinstance(model.yolo.names, dict):
            names = model.yolo.names
        elif isinstance(model.yolo.names, (list, tuple)):
            names = {i: n for i, n in enumerate(model.yolo.names)}

    # ------------------------
    # Draw + Save
    # ------------------------
    output = draw_boxes(img0.copy(), boxes, scores, classes, names)
    save_path = os.path.splitext(image_path)[0] + "_pred.jpg"
    cv2.imwrite(save_path, output)

    print("Saved:", save_path)
    '''
    print("Detections:")
    for c, s in zip(classes, scores):
        print(f" - {names.get(int(c), str(int(c)))} ({float(s):.2f})")
    '''
    print("Detections:")

    h, w = img0.shape[:2]  

    for box, c, s in zip(boxes, classes, scores):

        # xyxy format (pixel coordinates)
        x1, y1, x2, y2 = box

        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        print(
            f"{int(c)} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f} "
            f"# {names.get(int(c), str(int(c)))} ({float(s):.2f})"
            )


if __name__ == "__main__":
    main()