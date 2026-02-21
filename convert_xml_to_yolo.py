import os
import glob
import xml.etree.ElementTree as ET

# ----------------------------
# Config
# ----------------------------
# Convert annotated XML files to YOLOv8 .txt label format.
# ----------------------------
CLASS_MAP = {
    "fire": 0,
    "smoke": 1,
}

XML_DIR = "Annotations"          # folder containing .xml files
OUT_LABEL_DIR = "labels"  # output YOLOv8 label .txt files
MAX_FILES = 100           # read/convert 100 XML files (sorted)

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Pascal VOC (xmin,ymin,xmax,ymax) -> YOLO (cx,cy,w,h) normalized."""
    xmin = clamp(float(xmin), 0.0, img_w - 1.0)
    xmax = clamp(float(xmax), 0.0, img_w - 1.0)
    ymin = clamp(float(ymin), 0.0, img_h - 1.0)
    ymax = clamp(float(ymax), 0.0, img_h - 1.0)

    # Ensure proper ordering
    x1, x2 = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
    y1, y2 = (ymin, ymax) if ymin <= ymax else (ymax, ymin)

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return cx / img_w, cy / img_h, bw / img_w, bh / img_h

def safe_text(node, default=""):
    return node.text.strip() if (node is not None and node.text) else default

def parse_voc_xml(xml_path):
    """
    Returns (base_name_for_output, img_w, img_h, objects)
    objects list: (cls_name, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # filename may contain spaces and extension; we use basename without extension for label file
    filename = safe_text(root.find("filename"), "")
    if filename:
        base_name = os.path.splitext(os.path.basename(filename))[0]
    else:
        base_name = os.path.splitext(os.path.basename(xml_path))[0]

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    img_w = float(safe_text(size.find("width")))
    img_h = float(safe_text(size.find("height")))

    objects = []
    for obj in root.findall("object"):
        cls_name = safe_text(obj.find("name"), "").lower()
        bnd = obj.find("bndbox")
        if not cls_name or bnd is None:
            continue

        xmin = safe_text(bnd.find("xmin"))
        ymin = safe_text(bnd.find("ymin"))
        xmax = safe_text(bnd.find("xmax"))
        ymax = safe_text(bnd.find("ymax"))

        if xmin == "" or ymin == "" or xmax == "" or ymax == "":
            continue

        objects.append((cls_name, xmin, ymin, xmax, ymax))

    return base_name, img_w, img_h, objects

# ----------------------------
# Convert XMLs
# ----------------------------
xml_files = sorted(glob.glob(os.path.join(XML_DIR, "*.xml")))[:MAX_FILES]
if not xml_files:
    raise FileNotFoundError(f"No XML files found in: {XML_DIR}")

converted = 0
empty_labels = 0
skipped_unknown = 0
errors = 0

for xml_path in xml_files:
    try:
        base_name, img_w, img_h, objects = parse_voc_xml(xml_path)
        out_txt = os.path.join(OUT_LABEL_DIR, base_name + ".txt")

        lines = []
        for cls_name, xmin, ymin, xmax, ymax in objects:
            if cls_name not in CLASS_MAP:
                skipped_unknown += 1
                continue

            cls_id = CLASS_MAP[cls_name]
            cx, cy, bw, bh = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

            # Skip degenerate boxes
            if bw <= 0 or bh <= 0:
                continue

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Write label file (YOLO expects an empty file if no objects)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        converted += 1
        if not lines:
            empty_labels += 1

    except Exception as e:
        errors += 1
        print(f"[ERROR] {xml_path}: {e}")

print("===================================")
print(f"XML folder        : {XML_DIR}")
print(f"Output labels     : {OUT_LABEL_DIR}")
print(f"Processed (max)   : {len(xml_files)}")
print(f"Converted         : {converted}")
print(f"Empty label files : {empty_labels}")
print(f"Unknown-class objs: {skipped_unknown}")
print(f"Errors            : {errors}")
print("===================================")