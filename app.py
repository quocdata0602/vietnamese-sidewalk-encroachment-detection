import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageOps
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sidewalk Encroachment Detection",
    layout="wide"
)

st.title("Hệ thống phát hiện và đánh giá lấn chiếm vỉa hè")
st.caption("Object Detection + Road/Sidewalk Segmentation + Encroachment Score")

MODEL_DIR = Path("models")
SAMPLE_DIR = Path("sample_images")
OUTPUT_DIR = Path("outputs")
TEMP_UPLOAD_DIR = Path("temp_uploads")

MODEL_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)

DEFAULT_DET_MODEL = MODEL_DIR / "best_object_detection.pt"
DEFAULT_SEG_MODEL = MODEL_DIR / "best_road_sidewalk_seg.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DET_IMGSZ = 1280
SEG_IMGSZ = 640
OBJECT_SCORE_SCALE = 0.5

# Các tham số cố định, không hiển thị trên giao diện
DEFAULT_MAX_OBJECT_SCORE = 3.0
DEFAULT_MAX_DET = 300
DEFAULT_SIDEWALK_CLASS_NAME = "sidewalk"

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_yaml_config(filename, default=None):
    config_path = CONFIG_DIR / filename
    if not config_path.exists():
        return default if default is not None else {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


contact_config = load_yaml_config("contact_thresholds.yaml", {})
duplicate_config = load_yaml_config("duplicate_detection.yaml", {})
scoring_config = load_yaml_config("scoring.yaml", {})
sidewalk_postprocess_config = load_yaml_config("sidewalk_postprocess.yaml", {})

# Kiểm tra vùng đáy bbox để biết object có thật sự đứng trên sidewalk không.
# Với xe máy/ô tô chỉ xét vùng đáy mỏng, tránh phần người/thân xe bị dính vào sidewalk.
CONTACT_BAND_RATIOS = contact_config.get("contact_band_ratios", {})
CONTACT_THRESHOLDS = contact_config.get("contact_thresholds", {})

# =========================
# DUPLICATE DETECTION SUPPRESSION CONFIG
# =========================
# Một số class như vendor_cart/table_chair/umbrella dễ bị YOLO bắt 2-3 bbox cho cùng 1 vật thể.
DUPLICATE_SUPPRESS_CLASSES = set(duplicate_config.get("duplicate_suppress_classes", []))
# IoU càng thấp thì càng mạnh tay gộp bbox trùng.
DUPLICATE_IOU_THRESHOLDS = duplicate_config.get("duplicate_iou_thresholds", {})

# Nếu bbox nhỏ nằm phần lớn trong bbox lớn thì xem là trùng.
DUPLICATE_CONTAINMENT_THRESHOLD = duplicate_config.get(
    "duplicate_containment_threshold",
    0.45,
)

# Tâm 2 bbox đủ gần thì mới xem là cùng một object.
DUPLICATE_CENTER_DISTANCE_THRESHOLD = duplicate_config.get(
    "duplicate_center_distance_threshold",
    0.55,
)

# Hậu xử lý mask để loại nhiễu nhỏ và khôi phục phần vỉa hè bị vật cản che.
SIDEWALK_MIN_COMPONENT_RATIO = sidewalk_postprocess_config.get(
    "sidewalk_min_component_ratio",
    0.001,
)
SIDEWALK_CLOSE_KERNEL_RATIO = sidewalk_postprocess_config.get(
    "sidewalk_close_kernel_ratio",
    0.01,
)
SIDEWALK_MAX_HULL_EXPANSION = sidewalk_postprocess_config.get(
    "sidewalk_max_hull_expansion",
    1.5,
)


# =========================
# SCORING CONFIG
# =========================
DEFAULT_CLASS_WEIGHTS = scoring_config.get("default_class_weights", {})
DEFAULT_CLASS_CONF_THRESHOLDS = scoring_config.get("default_class_conf_thresholds", {})

# =========================
# LOAD MODEL
# =========================
@st.cache_resource(show_spinner="Đang load model...")
def load_model(model_path):
    return YOLO(str(model_path))


# =========================
# BASIC UTILS
# =========================
def normalize_class_name(name):
    return str(name).lower().strip()


def fmt_num(x):
    try:
        x = float(x)
        if x.is_integer():
            return str(int(x))
        return f"{x:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def coverage_score_from_ratio(ratio):
    if ratio <= 0:
        return 0
    if ratio < 0.05:
        return 0.25
    if ratio < 0.20:
        return 0.5
    if ratio < 0.40:
        return 1
    return 2


def severity_from_score(score):
    if score < 0.42:
        return "Không lấn chiếm"
    if score <= 1.50:
        return "Lấn chiếm nhẹ"
    if score <= 3.00:
        return "Lấn chiếm vừa"
    return "Lấn chiếm nặng"


def clear_folder(folder_path):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    for p in folder_path.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def save_uploaded_image(uploaded_file):
    clear_folder(TEMP_UPLOAD_DIR)

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = ".jpg"

    save_path = TEMP_UPLOAD_DIR / f"current_upload{suffix}"

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = ImageOps.exif_transpose(Image.open(save_path)).convert("RGB")
    img.save(save_path, quality=95)

    return save_path


def read_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    return image


def list_sample_images():
    sample_images = []
    for ext in IMAGE_EXTS:
        sample_images.extend(SAMPLE_DIR.glob(f"*{ext}"))
    return sorted(sample_images)


# =========================
# SIDEWALK MASK
# =========================
def postprocess_sidewalk_mask(sidewalk_mask, fill_occlusions=True):
    """Làm sạch mask và tùy chọn lấp các phần vỉa hè bị che khuất."""
    mask = (sidewalk_mask > 0).astype(np.uint8)
    h, w = mask.shape

    if mask.sum() == 0:
        return mask

    # Bỏ các mảng rời quá nhỏ do model dự đoán nhầm trên lòng đường.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )
    min_component_area = max(
        64,
        int(h * w * SIDEWALK_MIN_COMPONENT_RATIO)
    )
    cleaned_mask = np.zeros_like(mask)

    for label_id in range(1, num_labels):
        component_area = int(stats[label_id, cv2.CC_STAT_AREA])
        if component_area >= min_component_area:
            cleaned_mask[labels == label_id] = 1

    # Không xóa toàn bộ mask nếu ảnh chỉ có một vùng vỉa hè nhỏ ở xa.
    if cleaned_mask.sum() == 0:
        cleaned_mask = mask

    # Đóng các khe hở nhỏ trên cùng một vùng vỉa hè.
    kernel_size = max(3, int(min(h, w) * SIDEWALK_CLOSE_KERNEL_RATIO))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    if not fill_occlusions:
        return cleaned_mask

    # Khôi phục hành lang vỉa hè ẩn dưới xe/bàn ghế bằng convex hull.
    # Giới hạn độ nở để tránh lấp tràn sang lòng đường khi mask quá lõm.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned_mask,
        connectivity=8
    )
    filled_mask = np.zeros_like(cleaned_mask)

    for label_id in range(1, num_labels):
        component = (labels == label_id).astype(np.uint8)
        component_area = int(stats[label_id, cv2.CC_STAT_AREA])
        points = cv2.findNonZero(component)

        if points is None or component_area <= 0:
            continue

        hull_mask = np.zeros_like(component)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        hull_area = int(hull_mask.sum())

        if hull_area <= component_area * SIDEWALK_MAX_HULL_EXPANSION:
            filled_mask = np.maximum(filled_mask, hull_mask)
        else:
            filled_mask = np.maximum(filled_mask, component)

    return filled_mask


def get_sidewalk_mask(
    image,
    seg_model,
    sidewalk_class_name="sidewalk",
    imgsz=640,
    conf=0.15,
    fill_occlusions=True,
    return_road=False,
    remove_road_overlap=True
):
    h, w = image.shape[:2]
    sidewalk_mask = np.zeros((h, w), dtype=np.uint8)
    road_mask = np.zeros((h, w), dtype=np.uint8)

    result = seg_model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        retina_masks=True,
        verbose=False
    )[0]

    if result.masks is None or result.boxes is None:
        if return_road:
            return sidewalk_mask, road_mask
        return sidewalk_mask

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = seg_model.names

    for mask, cls_id in zip(masks, classes):
        cls_name = normalize_class_name(names[int(cls_id)])

        if cls_name in {sidewalk_class_name.lower().strip(), "road"}:
            if mask.shape == (h, w):
                mask_resized = mask
            else:
                mask_resized = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

        if cls_name == sidewalk_class_name.lower().strip():
            sidewalk_mask = np.maximum(sidewalk_mask, mask_binary)
        elif cls_name == "road":
            road_mask = np.maximum(road_mask, mask_binary)

    sidewalk_mask = postprocess_sidewalk_mask(
        sidewalk_mask,
        fill_occlusions=fill_occlusions
    )

    # Với inference thông thường, ưu tiên road ở vùng hai mask chồng nhau.
    # Khi tạo fixed mask, có thể giữ nguyên phần chồng lấn để giải quyết bằng vote
    # trên nhiều frame thay vì để một frame road xóa sidewalk ngay lập tức.
    if remove_road_overlap:
        sidewalk_mask[road_mask == 1] = 0

    if return_road:
        return sidewalk_mask, road_mask

    return sidewalk_mask


# =========================
# VISUALIZATION
# =========================
def draw_label(image, text, x, y, color, font_scale=0.55, thickness=2):
    h, w = image.shape[:2]
    y = max(22, int(y))
    x = max(0, min(int(x), w - 1))

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x2 = min(w - 1, x + tw + 8)
    y1 = max(0, y - th - baseline - 8)
    y2 = min(h - 1, y + 4)

    cv2.rectangle(image, (x, y1), (x2, y2), color, -1)
    cv2.putText(
        image,
        text,
        (x + 4, y - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )


def overlay_sidewalk(image, sidewalk_mask):
    vis = image.copy()
    overlay = vis.copy()

    # BGR: vàng
    overlay[sidewalk_mask == 1] = (0, 255, 255)
    vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    return vis


def draw_summary_panel(
    image,
    encroachment_score,
    severity,
    object_score,
    coverage_score,
    encroachment_count,
    coverage_ratio
):
    h, w = image.shape[:2]

    info1 = (
        f"Score: {encroachment_score:.2f} | "
        f"Severity: {severity} | "
        f"Objects: {encroachment_count} | "
        f"Coverage: {coverage_ratio:.2%}"
    )

    info2 = (
        f"Object score: {fmt_num(object_score)} | "
        f"Coverage score: {fmt_num(coverage_score)}"
    )

    panel_w = min(w - 20, 1250)
    cv2.rectangle(image, (10, 10), (10 + panel_w, 86), (0, 0, 0), -1)

    cv2.putText(
        image,
        info1,
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        info2,
        (22, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2
    )

def compute_contact_overlap(sidewalk_mask, road_mask, x1, y1, x2, y2, cls_name):
    """
    Tính vùng đáy bbox nằm trên sidewalk và road.
    Mục tiêu: tránh lỗi xe chạy trên road nhưng phần trên bbox dính sidewalk.
    """
    h, w = sidewalk_mask.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0, 0.0, 0.0, 0.0

    bbox_h = y2 - y1
    bbox_w = x2 - x1

    band_ratio = CONTACT_BAND_RATIOS.get(cls_name, 0.35)

    contact_y1 = int(y2 - bbox_h * band_ratio)
    contact_y1 = max(y1, min(contact_y1, y2 - 1))

    contact_sidewalk = sidewalk_mask[contact_y1:y2, x1:x2]
    contact_road = road_mask[contact_y1:y2, x1:x2]

    contact_total_area = int(contact_sidewalk.shape[0] * contact_sidewalk.shape[1])

    if contact_total_area > 0:
        contact_sidewalk_ratio = int(contact_sidewalk.sum()) / contact_total_area
        contact_road_ratio = int(contact_road.sum()) / contact_total_area
    else:
        contact_sidewalk_ratio = 0.0
        contact_road_ratio = 0.0

    contact_area = int(contact_sidewalk.sum())

    # Vùng đáy giữa bbox
    cx = int((x1 + x2) / 2)
    patch_w = max(4, int(bbox_w * 0.12))
    patch_h = max(4, int(bbox_h * 0.10))

    px1 = max(0, cx - patch_w // 2)
    px2 = min(w, cx + patch_w // 2)
    py1 = max(0, y2 - patch_h)
    py2 = min(h, y2)

    bottom_sidewalk = sidewalk_mask[py1:py2, px1:px2]
    bottom_road = road_mask[py1:py2, px1:px2]

    patch_area = int(bottom_sidewalk.shape[0] * bottom_sidewalk.shape[1])

    if patch_area > 0:
        bottom_center_sidewalk_ratio = int(bottom_sidewalk.sum()) / patch_area
        bottom_center_road_ratio = int(bottom_road.sum()) / patch_area
    else:
        bottom_center_sidewalk_ratio = 0.0
        bottom_center_road_ratio = 0.0

    return (
        float(contact_sidewalk_ratio),
        int(contact_area),
        float(bottom_center_sidewalk_ratio),
        float(contact_road_ratio),
        float(bottom_center_road_ratio),
    )


def bbox_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, float(ix2 - ix1))
    inter_h = max(0.0, float(iy2 - iy1))
    inter = inter_w * inter_h

    area_a = bbox_area_xyxy(box_a)
    area_b = bbox_area_xyxy(box_b)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def bbox_containment_xyxy(box_a, box_b):
    """
    Tỷ lệ phần giao / diện tích bbox nhỏ hơn.
    Nếu bbox nhỏ gần như nằm trong bbox lớn thì giá trị cao.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, float(ix2 - ix1))
    inter_h = max(0.0, float(iy2 - iy1))
    inter = inter_w * inter_h

    area_a = bbox_area_xyxy(box_a)
    area_b = bbox_area_xyxy(box_b)
    min_area = min(area_a, area_b)

    return inter / min_area if min_area > 0 else 0.0


def bbox_center_distance_norm(box_a, box_b):
    """
    Khoảng cách tâm bbox, chuẩn hóa theo đường chéo bbox lớn hơn.
    Giá trị càng nhỏ thì 2 bbox càng gần nhau.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)

    diag = max((aw ** 2 + ah ** 2) ** 0.5, (bw ** 2 + bh ** 2) ** 0.5)

    return dist / diag if diag > 0 else 999.0


def is_duplicate_detection(box_a, box_b, cls_name):
    """
    Kiểm tra 2 bbox cùng class có phải đang bắt trùng cùng một object không.
    Dùng thêm containment và khoảng cách tâm vì vendor_cart đôi khi có IoU không quá cao
    nhưng vẫn là cùng một xe hàng.
    """
    iou = bbox_iou_xyxy(box_a, box_b)
    containment = bbox_containment_xyxy(box_a, box_b)
    center_dist = bbox_center_distance_norm(box_a, box_b)

    iou_threshold = DUPLICATE_IOU_THRESHOLDS.get(cls_name, 0.30)

    if iou >= iou_threshold:
        return True

    if (
        containment >= DUPLICATE_CONTAINMENT_THRESHOLD
        and center_dist <= DUPLICATE_CENTER_DISTANCE_THRESHOLD
    ):
        return True

    return False


def make_duplicate_suppressed_row(box, cls_name, conf):
    x1, y1, x2, y2 = box.astype(int)
    return {
        "class": cls_name,
        "confidence": round(float(conf), 4),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "weight": 0,
        "sidewalk_overlap_ratio": 0,
        "sidewalk_overlap_percent": 0,
        "intersection_area_px": 0,
        "contact_area_px": 0,
        "contact_overlap_ratio": 0,
        "contact_overlap_percent": 0,
        "bottom_center_sidewalk_ratio": 0,
        "bottom_center_sidewalk_percent": 0,
        "contact_road_ratio": 0,
        "contact_road_percent": 0,
        "bottom_center_road_ratio": 0,
        "bottom_center_road_percent": 0,
        "is_encroachment": False,
        "reason": f"duplicate_suppressed_same_{cls_name}",
    }


def suppress_duplicate_detections(boxes, classes, confs, det_names):
    """
    Lọc bbox trùng sau YOLO NMS.
    Giữ lại bbox tốt nhất cho các class dễ bị trùng như vendor_cart.
    BBox bị loại vẫn được đưa vào ignored_objects để debug, nhưng không cộng score.
    """
    if boxes is None or len(boxes) == 0:
        return boxes, classes, confs, []

    areas = np.array([bbox_area_xyxy(b) for b in boxes], dtype=float)
    max_area = max(float(areas.max()), 1.0)

    # Ưu tiên confidence, cộng nhẹ diện tích để tránh giữ bbox quá nhỏ.
    priority = confs + 0.10 * (areas / max_area)
    order = np.argsort(-priority)

    keep_indices = []
    suppressed_rows = []

    for idx in order:
        idx = int(idx)
        cls_id = int(classes[idx])
        cls_name = normalize_class_name(det_names[cls_id])

        duplicate_of = None

        if cls_name in DUPLICATE_SUPPRESS_CLASSES:
            for kept_idx in keep_indices:
                kept_cls_id = int(classes[kept_idx])
                kept_cls_name = normalize_class_name(det_names[kept_cls_id])

                if cls_name != kept_cls_name:
                    continue

                if is_duplicate_detection(boxes[idx], boxes[kept_idx], cls_name):
                    duplicate_of = kept_idx
                    break

        if duplicate_of is None:
            keep_indices.append(idx)
        else:
            suppressed_rows.append(
                make_duplicate_suppressed_row(
                    box=boxes[idx],
                    cls_name=cls_name,
                    conf=confs[idx]
                )
            )

    keep_indices = np.array(keep_indices, dtype=int)

    return (
        boxes[keep_indices],
        classes[keep_indices],
        confs[keep_indices],
        suppressed_rows
    )


# =========================
# CORE IMAGE PIPELINE
# =========================
def process_frame(
    image,
    det_model,
    seg_model,
    det_conf=0.20,
    det_iou=0.50,
    seg_conf=0.20,
    sidewalk_overlap_threshold=0.05,
    sidewalk_class_name="sidewalk",
    class_weights=None,
    class_conf_thresholds=None,
    max_object_score=3.0,
    max_det=300,
    agnostic_nms=True,
    draw_ignored=False,
    fill_sidewalk_occlusions=True
):
    h, w = image.shape[:2]

    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS

    if class_conf_thresholds is None:
        class_conf_thresholds = DEFAULT_CLASS_CONF_THRESHOLDS

    sidewalk_mask, road_mask = get_sidewalk_mask(
        image=image,
        seg_model=seg_model,
        sidewalk_class_name=sidewalk_class_name,
        imgsz=SEG_IMGSZ,
        conf=seg_conf,
        fill_occlusions=fill_sidewalk_occlusions,
        return_road=True
    )
    mask_source = "image_segmentation"

    sidewalk_area = int(sidewalk_mask.sum())
    vis = overlay_sidewalk(image, sidewalk_mask)

    det_result = det_model.predict(
        source=image,
        imgsz=DET_IMGSZ,
        conf=det_conf,
        iou=det_iou,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        verbose=False
    )[0]

    det_names = det_model.names

    valid_objects = []
    ignored_objects = []
    detected_objects = []

    raw_object_score = 0.0
    obstacle_union_on_sidewalk = np.zeros((h, w), dtype=np.uint8)

    if det_result.boxes is not None:
        boxes = det_result.boxes.xyxy.cpu().numpy()
        classes = det_result.boxes.cls.cpu().numpy().astype(int)
        confs = det_result.boxes.conf.cpu().numpy()

        boxes, classes, confs, duplicate_rows = suppress_duplicate_detections(
            boxes=boxes,
            classes=classes,
            confs=confs,
            det_names=det_names
        )

        for dup_row in duplicate_rows:
            detected_objects.append(dup_row)
            ignored_objects.append(dup_row)

        for box, cls_id, score in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            cls_name = normalize_class_name(det_names[int(cls_id)])
            confidence = float(score)

            base_obj = {
                "class": cls_name,
                "confidence": round(confidence, 4),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }

            reason = None
            overlap_ratio = 0.0
            intersection_area = 0

            contact_overlap_ratio = 0.0
            contact_area = 0
            bottom_center_ratio = 0.0
            contact_road_ratio = 0.0
            bottom_center_road_ratio = 0.0

            weight = class_weights.get(cls_name, 1.0)

            if x2 <= x1 or y2 <= y1:
                reason = "invalid_bbox"
            else:
                min_conf = class_conf_thresholds.get(cls_name, det_conf)

                if confidence < min_conf:
                    reason = f"confidence < class_threshold ({min_conf})"
                elif sidewalk_area <= 0:
                    reason = "no_sidewalk_detected"
                else:
                    bbox_area = int((x2 - x1) * (y2 - y1))

                    # 1. Overlap toàn bbox với sidewalk
                    bbox_sidewalk_mask = sidewalk_mask[y1:y2, x1:x2]
                    intersection_area = int(bbox_sidewalk_mask.sum())
                    overlap_ratio = intersection_area / bbox_area if bbox_area > 0 else 0

                    # 2. Overlap vùng đáy bbox với sidewalk
                    # Dùng để tránh lỗi xe chạy trên road nhưng phần trên bbox dính sidewalk
                    (
                        contact_overlap_ratio,
                        contact_area,
                        bottom_center_ratio,
                        contact_road_ratio,
                        bottom_center_road_ratio,
                    ) = compute_contact_overlap(
                        sidewalk_mask=sidewalk_mask,
                        road_mask=road_mask,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        cls_name=cls_name
                    )

                    min_contact = CONTACT_THRESHOLDS.get(
                        cls_name,
                        max(0.05, sidewalk_overlap_threshold)
                    )

                    if overlap_ratio < sidewalk_overlap_threshold:
                        reason = f"sidewalk_overlap < {sidewalk_overlap_threshold}"

                    elif cls_name in {"car", "motorbike"}:
                        min_vehicle_contact = CONTACT_THRESHOLDS.get(cls_name, 0.22)

                        min_vehicle_bottom = {
                            "car": 0.35,
                            "motorbike": 0.30,
                        }.get(cls_name, 0.30)

                        # Xe được xem là có tiếp xúc với sidewalk khi:
                        # 1. Bottom-center đạt ngưỡng
                        # HOẶC
                        # 2. Lower contact band đạt ngưỡng
                        has_sidewalk_contact = (
                            bottom_center_ratio >= min_vehicle_bottom
                            or contact_overlap_ratio >= min_vehicle_contact
                        )

                        # Chỉ loại khi cả hai nguồn bằng chứng đều không đạt.
                        if not has_sidewalk_contact:
                            reason = (
                                f"vehicle_contact_not_on_sidewalk "
                                f"(bottom_sidewalk={bottom_center_ratio:.3f}, "
                                f"contact_sidewalk={contact_overlap_ratio:.3f}, "
                                f"bottom_road={bottom_center_road_ratio:.3f}, "
                                f"contact_road={contact_road_ratio:.3f})"
                            )

                        # Khi đã có bằng chứng tiếp xúc sidewalk,
                        # tiếp tục loại xe có road contact lớn hơn sidewalk contact.
                        elif contact_road_ratio > contact_overlap_ratio:
                            reason = (
                                f"vehicle_more_on_road_than_sidewalk "
                                f"(road_contact={contact_road_ratio:.3f}, "
                                f"sidewalk_contact={contact_overlap_ratio:.3f})"
                            )

                    elif contact_overlap_ratio < min_contact and bottom_center_ratio < 0.30:
                        reason = (
                            f"contact_overlap < {min_contact} "
                            f"(contact={contact_overlap_ratio:.3f}, bottom={bottom_center_ratio:.3f})"
                        )           


            obj_row = {
                **base_obj,
                "weight": weight,
                "sidewalk_overlap_ratio": round(float(overlap_ratio), 4),
                "sidewalk_overlap_percent": round(float(overlap_ratio) * 100, 2),
                "intersection_area_px": int(intersection_area),
                "contact_area_px": int(contact_area),
                "contact_overlap_ratio": round(float(contact_overlap_ratio), 4),
                "contact_overlap_percent": round(float(contact_overlap_ratio) * 100, 2),
                "bottom_center_sidewalk_ratio": round(float(bottom_center_ratio), 4),
                "bottom_center_sidewalk_percent": round(float(bottom_center_ratio) * 100, 2),
                "contact_road_ratio": round(float(contact_road_ratio), 4),
                "contact_road_percent": round(float(contact_road_ratio) * 100, 2),
                "bottom_center_road_ratio": round(float(bottom_center_road_ratio), 4),
                "bottom_center_road_percent": round(float(bottom_center_road_ratio) * 100, 2),
                "is_encroachment": reason is None,
                "reason": "" if reason is None else reason,
            }

            detected_objects.append(obj_row)

            if reason is None:
                raw_object_score += (
                    OBJECT_SCORE_SCALE
                    * weight
                    * confidence
                    * overlap_ratio
                )

                obstacle_union_on_sidewalk[y1:y2, x1:x2] = np.maximum(
                    obstacle_union_on_sidewalk[y1:y2, x1:x2],
                    sidewalk_mask[y1:y2, x1:x2]
                )

                valid_objects.append(obj_row)

                color = (0, 0, 255)
                label = (
                    f"{cls_name} {confidence:.2f} | "
                    f"ov {overlap_ratio * 100:.1f}% | "
                    f"ct {contact_overlap_ratio * 100:.1f}% | "
                    f"bt {bottom_center_ratio * 100:.1f}%"
                )
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 7)
                draw_label(vis, label, x1, y1 - 8, color)

            else:
                ignored_objects.append(obj_row)

                if draw_ignored:
                    color = (120, 120, 120)
                    label = f"{cls_name} {confidence:.2f} | ignored"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 5)
                    draw_label(vis, label, x1, y1 - 8, color)

    obstacle_area_on_sidewalk = int(obstacle_union_on_sidewalk.sum())
    coverage_ratio = obstacle_area_on_sidewalk / sidewalk_area if sidewalk_area > 0 else 0
    coverage_percent = coverage_ratio * 100

    object_score = min(float(raw_object_score), float(max_object_score))
    coverage_score = coverage_score_from_ratio(coverage_ratio)
    encroachment_score = object_score + coverage_score
    severity = severity_from_score(encroachment_score)

    draw_summary_panel(
        image=vis,
        encroachment_score=encroachment_score,
        severity=severity,
        object_score=object_score,
        coverage_score=coverage_score,
        encroachment_count=len(valid_objects),
        coverage_ratio=coverage_ratio
    )

    score_row = {
        "num_obstacle_regions_on_sidewalk": len(valid_objects),
        "raw_object_score": round(raw_object_score, 2),
        "object_score": round(object_score, 2),
        "mask_source": mask_source,
        "sidewalk_detected": bool(sidewalk_area > 0),
        "sidewalk_area_px": sidewalk_area,
        "sidewalk_area_ratio": round(sidewalk_area / max(1, h * w), 6),
        "obstacle_area_on_sidewalk_px": obstacle_area_on_sidewalk,
        "coverage_ratio": round(coverage_ratio, 4),
        "coverage_percent": round(coverage_percent, 2),
        "coverage_score": coverage_score,
        "encroachment_score": round(encroachment_score, 2),
        "severity_level": severity,
    }

    return {
        "vis": vis,
        "severity": severity,
        "encroachment_score": encroachment_score,
        "encroachment_count": len(valid_objects),
        "raw_object_score": raw_object_score,
        "object_score": object_score,
        "coverage_score": coverage_score,
        "coverage_ratio": coverage_ratio,
        "coverage_percent": coverage_percent,
        "sidewalk_area": sidewalk_area,
        "obstacle_area_on_sidewalk": obstacle_area_on_sidewalk,
        "detected_objects": detected_objects,
        "used_objects": valid_objects,
        "ignored_objects": ignored_objects,
        "score_row": score_row,
    }


# =========================
# IMAGE PIPELINE
# =========================
def run_image_pipeline(
    image_path,
    det_model,
    seg_model,
    det_conf=0.20,
    det_iou=0.50,
    seg_conf=0.2,
    sidewalk_overlap_threshold=0.05,
    sidewalk_class_name="sidewalk",
    class_weights=None,
    class_conf_thresholds=None,
    max_object_score=3.0,
    max_det=300,
    agnostic_nms=True,
    draw_ignored=False,
    fill_sidewalk_occlusions=True
):
    image = read_image(image_path)

    result = process_frame(
        image=image,
        det_model=det_model,
        seg_model=seg_model,
        det_conf=det_conf,
        det_iou=det_iou,
        seg_conf=seg_conf,
        sidewalk_overlap_threshold=sidewalk_overlap_threshold,
        sidewalk_class_name=sidewalk_class_name,
        class_weights=class_weights,
        class_conf_thresholds=class_conf_thresholds,
        max_object_score=max_object_score,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        draw_ignored=draw_ignored,
        fill_sidewalk_occlusions=fill_sidewalk_occlusions
    )

    stem = Path(image_path).stem
    output_path = OUTPUT_DIR / f"pipeline_{stem}.jpg"
    cv2.imwrite(str(output_path), result["vis"])

    score_row = {
        "image_name": Path(image_path).name,
        **result["score_row"],
        "result_image_path": str(output_path),
    }

    score_df = pd.DataFrame([score_row])
    used_df = pd.DataFrame(result["used_objects"])
    ignored_df = pd.DataFrame(result["ignored_objects"])
    all_df = pd.DataFrame(result["detected_objects"])

    score_csv = OUTPUT_DIR / f"{stem}_encroachment_score_results.csv"
    used_csv = OUTPUT_DIR / f"{stem}_obstacle_predictions_used.csv"
    ignored_csv = OUTPUT_DIR / f"{stem}_obstacle_predictions_ignored.csv"
    all_csv = OUTPUT_DIR / f"{stem}_all_predictions.csv"

    score_df.to_csv(score_csv, index=False, encoding="utf-8-sig")
    used_df.to_csv(used_csv, index=False, encoding="utf-8-sig")
    ignored_df.to_csv(ignored_csv, index=False, encoding="utf-8-sig")
    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    return {
        **result,
        "output_path": output_path,
        "score_row": score_row,
        "score_csv": score_csv,
        "used_csv": used_csv,
        "ignored_csv": ignored_csv,
        "all_csv": all_csv,
    }


# =========================
# SIDEBAR - GỌN HƠN
# =========================
st.sidebar.header("Cấu hình nhanh")
st.sidebar.caption(
    f"Image size: detection {DET_IMGSZ} | segmentation {SEG_IMGSZ}"
)

det_conf = st.sidebar.slider(
    "Detection confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.20,
    step=0.05,
    help=(
        "Ngưỡng tin cậy chung của model phát hiện vật thể. "
        "Tăng giá trị này để giảm false positive, nhưng có thể bỏ sót vật thể nhỏ hoặc mờ."
    )
)

seg_conf = st.sidebar.slider(
    "Segmentation confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.25,
    step=0.05,
    help=(
        "Ngưỡng tin cậy của model phân đoạn road/sidewalk. "
        "Tăng giá trị này giúp mask chắc hơn, nhưng có thể làm mất bớt vùng sidewalk."
    )
)

sidewalk_overlap_threshold = st.sidebar.slider(
    "Sidewalk overlap threshold",
    min_value=0.00,
    max_value=0.50,
    value=0.05,
    step=0.01,
    help=(
        "Tỷ lệ tối thiểu giữa bbox vật thể và vùng sidewalk để vật thể được xem xét là lấn chiếm. "
        "Giảm giá trị này để bắt nhiều vật thể hơn; tăng giá trị này để lọc các vật thể chỉ dính mép sidewalk."
    )
)

# Các tham số này vẫn dùng trong pipeline nhưng không hiển thị trên Advanced settings.
max_object_score = DEFAULT_MAX_OBJECT_SCORE
max_det = DEFAULT_MAX_DET
sidewalk_class_name = DEFAULT_SIDEWALK_CLASS_NAME


with st.sidebar.expander("Advanced settings"):
    det_model_path = st.text_input(
        "Object detection model",
        value=str(DEFAULT_DET_MODEL),
        help=(
            "Đường dẫn tới file model YOLO dùng để phát hiện vật thể lấn chiếm, "
            "ví dụ: models/best_object_detection.pt."
        )
    )

    seg_model_path = st.text_input(
        "Road/Sidewalk segmentation model",
        value=str(DEFAULT_SEG_MODEL),
        help=(
            "Đường dẫn tới file model YOLO segmentation dùng để phân đoạn road và sidewalk, "
            "ví dụ: models/best_road_sidewalk_seg.pt."
        )
    )

    det_iou = st.slider(
        "Detection IoU",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        key="det_iou_v2",
        help=(
            "Ngưỡng IoU dùng trong Non-Maximum Suppression của YOLO. "
            "Giảm giá trị này nếu một vật thể bị nhiều bbox trùng nhau; tăng nếu bbox bị loại quá mạnh."
        )
    )

    agnostic_nms = st.checkbox(
        "Agnostic NMS",
        value=True,
        help=(
            "Nếu bật, NMS sẽ loại bbox trùng nhau mà không phân biệt class. "
            "Nên bật khi model hay detect một vật thể thành nhiều class hoặc nhiều bbox."
        )
    )

    draw_ignored = st.checkbox(
        "Vẽ bbox bị bỏ qua",
        value=False,
        help=(
            "Nếu bật, hệ thống sẽ vẽ cả các bbox bị loại khỏi tính điểm. "
            "Dùng để debug lý do bbox bị bỏ qua như confidence thấp, không nằm trên sidewalk hoặc bị suppress trùng."
        )
    )

    fill_sidewalk_occlusions = st.checkbox(
        "Lấp vùng vỉa hè bị vật cản che",
        value=True,
        help=(
            "Khôi phục vùng sidewalk bị che bởi xe, bàn ghế hoặc vật cản. "
            "Có thể tắt nếu thao tác này làm vùng sidewalk nở quá mức xuống lòng đường."
        )
    )


with st.sidebar.expander("Ngưỡng confidence riêng từng class"):
    class_conf_thresholds = {}

    class_conf_help = {
        "ad_board": "Ngưỡng confidence riêng cho bảng hiệu/biển quảng cáo.",
        "bin": "Ngưỡng confidence riêng cho thùng rác.",
        "car": "Ngưỡng confidence riêng cho ô tô. Nên để cao hơn nếu xe trên road bị bắt nhầm là lấn chiếm.",
        "motorbike": "Ngưỡng confidence riêng cho xe máy. Tăng giá trị này nếu xe máy bị false positive nhiều.",
        "table_chair": "Ngưỡng confidence riêng cho bàn ghế. Tăng nếu bị nhận nhầm nhiều.",
        "umbrella": "Ngưỡng confidence riêng cho ô/dù.",
        "vendor_cart": "Ngưỡng confidence riêng cho xe bán hàng. Tăng nếu vendor_cart bị false positive nhiều."
    }

    for cls, val in DEFAULT_CLASS_CONF_THRESHOLDS.items():
        class_conf_thresholds[cls] = st.slider(
            cls,
            min_value=0.05,
            max_value=0.90,
            value=float(val),
            step=0.05,
            key=f"conf_threshold_v2_{cls}",
            help=class_conf_help.get(
                cls,
                "Ngưỡng confidence riêng cho class này."
            )
        )


with st.sidebar.expander("Trọng số từng class"):
    class_weights = {}

    class_weight_help = {
        "ad_board": "Trọng số ảnh hưởng của bảng hiệu/biển quảng cáo đến object score.",
        "bin": "Trọng số ảnh hưởng của thùng rác đến object score.",
        "car": "Trọng số ảnh hưởng của ô tô đến object score.",
        "motorbike": "Trọng số ảnh hưởng của xe máy đến object score.",
        "table_chair": "Trọng số ảnh hưởng của bàn ghế đến object score.",
        "umbrella": "Trọng số ảnh hưởng của ô/dù đến object score.",
        "vendor_cart": "Trọng số ảnh hưởng của xe bán hàng đến object score. Giá trị cao làm vendor_cart tác động mạnh hơn đến mức độ lấn chiếm."
    }

    for cls, val in DEFAULT_CLASS_WEIGHTS.items():
        class_weights[cls] = st.number_input(
            cls,
            min_value=0.0,
            max_value=5.0,
            value=float(val),
            step=0.1,
            key=f"weight_{cls}",
            help=class_weight_help.get(
                cls,
                "Trọng số ảnh hưởng của class này đến object score."
            )
        )

# =========================
# INPUT
# =========================
st.subheader("Chọn đầu vào")

input_mode = st.radio(
    "Nguồn đầu vào",
    ["Upload ảnh", "Chọn ngẫu nhiên từ sample_images"],
    horizontal=True
)

image_path = None

if input_mode == "Upload ảnh":
    uploaded_file = st.file_uploader(
        "Tải ảnh bất kỳ",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded_file is not None:
        image_path = save_uploaded_image(uploaded_file)

elif input_mode == "Chọn ngẫu nhiên từ sample_images":
    sample_images = list_sample_images()

    if len(sample_images) == 0:
        st.warning("Folder sample_images chưa có ảnh.")
    else:
        if st.button("Chọn ảnh ngẫu nhiên"):
            st.session_state["random_image"] = str(random.choice(sample_images))

        if "random_image" in st.session_state:
            image_path = Path(st.session_state["random_image"])



# =========================
# RUN IMAGE APP
# =========================
if image_path is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Ảnh gốc")
        st.image(Image.open(image_path), use_container_width=True)

    run_button = st.button("Chạy pipeline", type="primary", use_container_width=True)

    if run_button:
        det_model_path = Path(det_model_path)
        seg_model_path = Path(seg_model_path)

        if not det_model_path.exists():
            st.error(f"Không tìm thấy file object detection model: {det_model_path}")
            st.stop()

        if not seg_model_path.exists():
            st.error(f"Không tìm thấy file segmentation model: {seg_model_path}")
            st.stop()

        with st.spinner("Đang chạy pipeline..."):
            det_model = load_model(det_model_path)
            seg_model = load_model(seg_model_path)

            result = run_image_pipeline(
                image_path=image_path,
                det_model=det_model,
                seg_model=seg_model,
                det_conf=det_conf,
                det_iou=det_iou,
                seg_conf=seg_conf,
                sidewalk_overlap_threshold=sidewalk_overlap_threshold,
                sidewalk_class_name=sidewalk_class_name,
                class_weights=class_weights,
                class_conf_thresholds=class_conf_thresholds,
                max_object_score=max_object_score,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                draw_ignored=draw_ignored,
                fill_sidewalk_occlusions=fill_sidewalk_occlusions
            )

        with col2:
            st.markdown("### Kết quả pipeline")
            st.image(Image.open(result["output_path"]), use_container_width=True)

        st.subheader("Tóm tắt kết quả")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mức độ", result["severity"])
        m2.metric("Tổng điểm", f"{result['encroachment_score']:.2f}")
        m3.metric("Số vật cản", result["encroachment_count"])
        m4.metric("Tỷ lệ che phủ", f"{result['coverage_percent']:.2f}%")

        with st.expander("Chi tiết tính điểm"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Object score", f"{result['object_score']:.2f}")
            c2.metric("Coverage score", fmt_num(result["coverage_score"]))
            c3.metric("Diện tích sidewalk", result["sidewalk_area"])
            c4.metric("Diện tích bị che", result["obstacle_area_on_sidewalk"])

            st.dataframe(pd.DataFrame([result["score_row"]]), use_container_width=True)

        with st.expander("Debug bbox"):
            st.markdown("#### BBox được dùng để tính điểm")
            if len(result["used_objects"]) == 0:
                st.info("Không có bbox nào được dùng để tính điểm.")
            else:
                st.dataframe(pd.DataFrame(result["used_objects"]), use_container_width=True)

            st.markdown("#### BBox bị bỏ qua")
            if len(result["ignored_objects"]) == 0:
                st.info("Không có bbox nào bị bỏ qua.")
            else:
                st.dataframe(pd.DataFrame(result["ignored_objects"]), use_container_width=True)

        st.subheader("Tải kết quả")

        d1, d2 = st.columns(2)

        with d1:
            with open(result["output_path"], "rb") as f:
                st.download_button(
                    label="Tải ảnh kết quả",
                    data=f,
                    file_name=Path(result["output_path"]).name,
                    mime="image/jpeg",
                    use_container_width=True
                )

        with d2:
            with open(result["score_csv"], "rb") as f:
                st.download_button(
                    label="Tải score CSV",
                    data=f,
                    file_name=Path(result["score_csv"]).name,
                    mime="text/csv",
                    use_container_width=True
                )

        with st.expander("Tải thêm CSV debug"):
            c1, c2, c3 = st.columns(3)

            with c1:
                with open(result["used_csv"], "rb") as f:
                    st.download_button(
                        label="Tải bbox dùng CSV",
                        data=f,
                        file_name=Path(result["used_csv"]).name,
                        mime="text/csv"
                    )

            with c2:
                with open(result["ignored_csv"], "rb") as f:
                    st.download_button(
                        label="Tải bbox bỏ qua CSV",
                        data=f,
                        file_name=Path(result["ignored_csv"]).name,
                        mime="text/csv"
                    )

            with c3:
                with open(result["all_csv"], "rb") as f:
                    st.download_button(
                        label="Tải toàn bộ bbox CSV",
                        data=f,
                        file_name=Path(result["all_csv"]).name,
                        mime="text/csv"
                    )



else:
    st.info("Hãy upload ảnh hoặc chọn ảnh ngẫu nhiên từ folder sample_images.")
