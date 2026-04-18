import os
from collections import Counter
import time
import datetime
import requests
import io
import pytz
import hashlib
import csv
import glob
from urllib.parse import urlparse

import cv2
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image as PILImage

app = Flask(__name__)

# --- CONFIGURATION ---
model_path = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(model_path)
names = model.model.names
print("MODEL CLASS NAMES:", names)

# Telegram Configuration
BOT_TOKEN = "7692550301:AAGPTJrcCDdXuitHRNQep85UhfSUcXMxNkc"
TARGET_CHAT_ID = "-4887423941"
ENABLE_TELEGRAM = True  # ตั้งเป็น True เมื่อต้องการส่ง Telegram

# Vehicle detection model (สำหรับนับรถในกรอบ accident)
# Try multiple path options
VEHICLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "testmo", "model11n", "yolov11n.pt"),
    os.path.join(os.path.dirname(__file__), "..", "testmo", "model11n", "yolov11n.pt"),
    "testmo/model11n/yolov11n.pt",
]
vehicle_model = None
for vpath in VEHICLE_MODEL_PATHS:
    if os.path.exists(vpath):
        try:
            vehicle_model = YOLO(vpath)
            print(f"✅ โหลดโมเดล vehicle สำเร็จ: {os.path.abspath(vpath)}")
            break
        except Exception as e:
            print(f"   โหลดโมเดล {vpath} ล้มเหลว: {e}")
            continue
if vehicle_model is None:
    print(f"⚠️  ไม่พบไฟล์โมเดล vehicle - จะใช้เฉพาะ accident detection")

TARGET_VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# ===== PERFORMANCE OPTIMIZATION SETTINGS =====
ACCIDENT_CLASS_NAME = "accident"
ACCIDENT_CONF_THRESHOLD = 0.8
PROCESS_EVERY_N = 2          # Process every 2nd frame (was 3) → smoother playback (15fps from 30fps)
TRACK_CONF = 0.55           # Increased from 0.45 → less false positives, faster
TRACK_IOU = 0.45
VEHICLE_DETECTION_CONF = 0.45
DETECT_VEHICLES_FULL_FRAME = True  # Enable full-frame detection (runs async in background)

LABEL_REMAP = {
    "non-accident": "car",
    "Non-Accident": "car",
    "non_accident": "car",
}

detected_objects_by_file = {}
accident_log_by_file = {}
all_detections_by_file = {}   # Store all detections for evaluation
inference_stats_by_file = {}  # Track inference time per file

ACCIDENT_FRAME_DIR = "accident_frames"
os.makedirs(ACCIDENT_FRAME_DIR, exist_ok=True)

DETECTION_FRAME_DIR = "detection_frames"
os.makedirs(DETECTION_FRAME_DIR, exist_ok=True)


# --- HELPER FUNCTIONS ---
def apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img


def send_telegram_alert_async(image_array, detected_class, conf, vehicle_counts, filename):
    """Send alert notification to Telegram (runs in background thread)"""
    if not ENABLE_TELEGRAM or not TARGET_CHAT_ID:
        return
    try:
        send_telegram_alert(image_array, detected_class, conf, vehicle_counts, filename)
    except Exception as e:
        print(f"⚠️  Telegram alert error: {e}")

def send_telegram_alert(image_array, detected_class, conf, vehicle_counts, filename):
    """Send alert notification to Telegram with image and vehicle counts"""
    if not ENABLE_TELEGRAM or not TARGET_CHAT_ID:
        return
    
    tz_TH = pytz.timezone('Asia/Bangkok')
    current_time = datetime.datetime.now(tz_TH).strftime("%Y-%m-%d %H:%M:%S")

    vehicle_info_str = ""
    if vehicle_counts:
        for v_name, v_count in vehicle_counts.items():
            vehicle_info_str += f"            {v_name}: {v_count} คัน\n"
    else:
        vehicle_info_str = "          ไม่พบยานพาหนะในกรอบ\n"

    try:
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb_image)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        caption = (
            f"🚨 แจ้งเตือนอุบัติเหตุ!\n"
            f"Class: {detected_class} ({conf:.2f}%)\n"
            f"{vehicle_info_str}"
            f"Time (TH): {current_time}\n"
            f"Video: {filename}\n"
            f"Device: Web Server"
        )

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        files_data = {'photo': ('alert.jpg', img_byte_arr, 'image/jpeg')}
        data = {'chat_id': TARGET_CHAT_ID, 'caption': caption}

        requests.post(url, files=files_data, data=data, timeout=5)
        print(f"✅ ส่งแจ้งเตือน Telegram สำเร็จ!")
    except Exception as e:
        print(f"❌ Error sending telegram: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():

    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    filename = secure_filename(file.filename)
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # --- สร้างไฟล์ GT อัตโนมัติถ้ายังไม่มี หรือถ้าว่าง ให้เติมตัวอย่าง ---
    base = os.path.splitext(filename)[0]
    gt_path = os.path.join("uploads", f"{base}_gt.txt")
    example_path = os.path.join("uploads", f"{base}_gt_example.txt")
    example_lines = [
        "# GT format per line: frame,x1,y1,x2,y2,label",
        "# ตัวอย่าง:",
        "15,100,50,420,300,accident",
    ]

    # ถ้าไม่มีไฟล์ GT หรือตัวไฟล์ว่าง ให้เขียนตัวอย่างลงไปเพื่อไม่ให้เป็นไฟล์ว่าง
    try:
        if not os.path.exists(gt_path) or os.path.getsize(gt_path) == 0:
            with open(gt_path, "w", encoding="utf-8") as gf:
                gf.write("\n".join(example_lines) + "\n")
    except Exception as _e:
        print(f"Failed creating GT file {gt_path}: {_e}")

    # เก็บไฟล์ตัวอย่างอธิบายรูปแบบไว้ให้ผู้ใช้เปิดดูได้
    if not os.path.exists(example_path):
        try:
            with open(example_path, "w", encoding="utf-8") as ef:
                ef.write("\n".join(example_lines) + "\n")
        except Exception as _e:
            print(f"Failed creating example GT file {example_path}: {_e}")
    # --- จบการสร้างไฟล์ GT อัตโนมัติ ---

    detected_objects_by_file[filename] = []
    accident_log_by_file[filename] = []
    return redirect(url_for("play_video", filename=filename))


@app.route("/upload_url", methods=["POST"])
def upload_video_from_url():
    """Upload video from URL"""
    print("🔗 URL upload function called")
    video_url = request.form.get("video_url", "").strip()
    print(f"📥 Received URL: {video_url}")
    
    if not video_url:
        print("❌ No URL provided")
        flash("กรุณาป้อน URL วิดีโอ", "error")
        return redirect(url_for("index"))
    
    try:
        # Validate URL (less strict)
        parsed = urlparse(video_url)
        if not parsed.scheme or not parsed.netloc:
            print(f"❌ Invalid URL format: {video_url}")
            flash("รูปแบบ URL ไม่ถูกต้อง กรุณาใส่ URL ที่ถูกต้อง", "error")
            return redirect(url_for("index"))
        
        print(f"✅ URL validation passed: {parsed.scheme}://{parsed.netloc}")
        
        # Generate filename from URL
        url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
        filename = f"url_{url_hash}.mp4"
        
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", filename)
        
        # Download video from URL
        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check content type (less strict)
        content_type = response.headers.get('content-type', '').lower()
        has_video_content_type = 'video' in content_type or 'application/octet-stream' in content_type
        has_video_extension = any(ext in video_url.lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav'])
        
        if not has_video_content_type and not has_video_extension:
            flash("URL อาจไม่ชี้ไปยังไฟล์วิดีโอ (ลองตรวจสอบ URL อีกครั้ง)", "warning")
            # Continue anyway, let OpenCV verify later
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Download progress: {progress:.1f}%", end='\r')
        
        print(f"\nDownloaded {downloaded} bytes to {file_path}")
        
        # Verify the downloaded file is a valid video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            os.remove(file_path)
            flash("ไฟล์ที่ดาวน์โหลดไม่ใช่วิดีโอที่ถูกต้อง", "error")
            return redirect(url_for("index"))
        cap.release()
        
        # --- สร้างไฟล์ GT อัตโนมัติ ---
        base = os.path.splitext(filename)[0]
        gt_path = os.path.join("uploads", f"{base}_gt.txt")
        example_path = os.path.join("uploads", f"{base}_gt_example.txt")
        example_lines = [
            "# GT format per line: frame,x1,y1,x2,y2,label",
            "# ตัวอย่าง:",
            "15,100,50,420,300,accident",
        ]

        try:
            if not os.path.exists(gt_path) or os.path.getsize(gt_path) == 0:
                with open(gt_path, "w", encoding="utf-8") as gf:
                    gf.write("\n".join(example_lines) + "\n")
        except Exception as _e:
            print(f"Failed creating GT file {gt_path}: {_e}")

        if not os.path.exists(example_path):
            try:
                with open(example_path, "w", encoding="utf-8") as ef:
                    ef.write("\n".join(example_lines) + "\n")
            except Exception as _e:
                print(f"Failed creating example GT file {example_path}: {_e}")
        
        detected_objects_by_file[filename] = []
        accident_log_by_file[filename] = []
        
        print(f"Video from URL ready for processing: {filename}")
        flash(f"ดาวน์โหลดวิดีโอจาก URL สำเร็จ: {filename}", "success")
        return redirect(url_for("play_video", filename=filename))
        
    except requests.exceptions.RequestException as e:
        flash(f"เกิดข้อผิดพลาดในการดาวน์โหลดวิดีโอ: {str(e)}", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"เกิดข้อผิดพลาดในการประมวลผล URL: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/detected_objects/<filename>")
def get_detected_objects(filename):
    detected_objects = detected_objects_by_file.get(filename, [])
    return jsonify(detected_objects)


def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def load_gt_for_video(filename):
    # support uploads/<filename>_gt.txt or uploads/<basename>_gt.txt or uploads/<basename>.gt
    base = os.path.splitext(filename)[0]
    candidates = [
        os.path.join("uploads", f"{base}_gt.txt"),
        os.path.join("uploads", f"{base}_gt.csv"),
        os.path.join("uploads", f"{base}.gt"),
        os.path.join("uploads", f"{base}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            gt = []
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    # skip empty rows and comment lines
                    if not row:
                        continue
                    # join then strip to allow CSV parsers that split commented lines
                    line = ",".join(row).strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [s.strip() for s in line.split(",")]
                    # expect: frame,x1,y1,x2,y2,label
                    if len(parts) < 5:
                        continue
                    # first field must be integer frame index
                    try:
                        frame = int(parts[0])
                    except Exception:
                        continue
                    try:
                        x1, y1, x2, y2 = map(int, parts[1:5])
                    except Exception:
                        continue
                    label = parts[5].strip() if len(parts) > 5 else "accident"
                    # apply remap like detections (case-insensitive)
                    label = LABEL_REMAP.get(label, LABEL_REMAP.get(label.lower(), label))
                    gt.append({"frame": frame, "box": [x1, y1, x2, y2], "label": label})
            # if no valid GT entries found, treat as not found
            return gt if gt else None
    return None
# --- เพิ่มฟังก์ชันคำนวณ AP (สำหรับคลาสเดียว เช่น "accident") ---
def compute_ap(preds, gts, iou_thr=0.5, class_name=ACCIDENT_CLASS_NAME):
    """
    Simple AP calculation:
    - preds: list of detections with keys ['frame','box','confidence','label']
    - gts: list of GT entries with keys ['frame','box','label']
    - match per-frame, one-to-one, by best IoU >= iou_thr
    Returns AP (0..1)
    """
    # filter preds for class and sort by confidence desc (case-insensitive comparison)
    preds_c = [p for p in preds if p.get("label", "").lower() == class_name.lower()]
    preds_c.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    # index GTs by frame and mark unmatched
    gt_by_frame = {}
    total_gts = 0
    for i, g in enumerate(gts):
        if g.get("label", "").lower() != class_name.lower():
            continue
        gt_by_frame.setdefault(g["frame"], []).append({"box": g["box"], "matched": False})
        total_gts += 1

    if total_gts == 0:
        return 0.0

    tps = []
    fps = []

    for p in preds_c:
        pf = p.get("frame")
        pbox = p.get("box")
        candidates = gt_by_frame.get(pf, [])
        best_iou = 0.0
        best_idx = None
        for idx, cg in enumerate(candidates):
            if cg["matched"]:
                continue
            val = iou(pbox, cg["box"])
            if val > best_iou:
                best_iou = val
                best_idx = idx
        if best_iou >= iou_thr and best_idx is not None:
            # true positive
            candidates[best_idx]["matched"] = True
            tps.append(1)
            fps.append(0)
        else:
            # false positive
            tps.append(0)
            fps.append(1)

    # cumulative sums
    cum_tp = []
    cum_fp = []
    s_tp = s_fp = 0
    for tp, fp in zip(tps, fps):
        s_tp += tp
        s_fp += fp
        cum_tp.append(s_tp)
        cum_fp.append(s_fp)

    precisions = []
    recalls = []
    for ct, cf in zip(cum_tp, cum_fp):
        prec = ct / (ct + cf) if (ct + cf) > 0 else 0.0
        rec = ct / total_gts
        precisions.append(prec)
        recalls.append(rec)

    # interpolate precision to get AP (VOC-like)
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


@app.route("/evaluate/<filename>")
def evaluate(filename):
    """
    Simple evaluation mode: count TP/FP by box color and also report IoU-based mAP
    and inference time (average ms per frame).
    Accept optional ?iou= to control IoU threshold used for AP and display.
    """
    try:
        iou_thr = float(request.args.get("iou", 0.5))
    except Exception:
        iou_thr = 0.5

    gts = load_gt_for_video(filename) or []
    preds = all_detections_by_file.get(filename, []) or []

    # existing color-count style metrics (kept for backward compatibility)
    total_gt = sum(1 for _ in gts) if gts is not None else 0
    dets = preds
    total_pred = len(dets)

    # ตามคำขอ: นับ TP +1 สำหรับทุกการตรวจจับ (ทุกค่า) และนับ FP +1 เมื่อกรอบเป็นสีเขียว (non-accident)
    TP = total_pred
    FP = sum(1 for d in dets if not d.get("is_accident", False))

    # count accident detections with confidence lower than threshold -> each increases FN by 1
    low_conf_threshold = 0.75
    low_conf_accidents = sum(
        1
        for d in dets
        if (d.get("label", "").lower() == ACCIDENT_CLASS_NAME.lower()) and (d.get("confidence", 0.0) < low_conf_threshold)
    )

    FN = max(0, total_gt - TP) + low_conf_accidents

    # frame-level TN as before
    frames_seen = set()
    for p in preds:
        frames_seen.add(p.get("frame"))
    for g in gts:
        frames_seen.add(g.get("frame"))

    TN = 0
    for fr in frames_seen:
        has_gt_pos = any((g["frame"] == fr and g["label"].lower() == ACCIDENT_CLASS_NAME.lower()) for g in gts)
        has_pred_pos = any((p["frame"] == fr and p.get("label", "").lower() == ACCIDENT_CLASS_NAME.lower()) for p in preds)
        if not has_gt_pos and not has_pred_pos:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- compute AP (mAP for single class 'accident') using IoU matching at iou_thr ---
    ap = compute_ap(preds, gts, iou_thr=iou_thr, class_name=ACCIDENT_CLASS_NAME)
    mAP = ap  # only one class here

    # --- inference time stats ---
    stats = inference_stats_by_file.get(filename, {"total_time": 0.0, "count": 0})
    avg_inference_ms = (stats["total_time"] / stats["count"] * 1000.0) if stats["count"] > 0 else None

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "mode": "color-count",
        "iou_threshold": iou_thr,
        "track_conf_threshold": TRACK_CONF,
        "low_conf_accidents": low_conf_accidents,
        "mAP": round(mAP, 4),
        "avg_inference_ms": round(avg_inference_ms, 3) if avg_inference_ms is not None else None,
    }
    return jsonify(metrics)


def detect_objects_from_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    accident_log = []
    detections = []  # เก็บทุกรายการ detection (ทุกคลาส ทุกเฟรม)

    # ensure stats entry
    inference_stats_by_file.setdefault(filename, {"total_time": 0.0, "count": 0})

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        # process every Nth frame (use 1 for testing)
        if (count % PROCESS_EVERY_N) != 0:
            continue

        # resize for inference / display
        frame = cv2.resize(frame, (1020, 600))
        
        # Apply CLAHE enhancement for better contrast
        frame_enhanced = apply_clahe(frame)

        # initialize per-frame variables
        frame_has_detection = False
        class_ids = []
        boxes = []
        confidences = []
        track_ids = []
        detected_objects = []
        found_accident_this_frame = None  # Store accident info for this frame

        try:
            # run tracker/detector (adjust conf/iou when tuning) - use enhanced frame
            t0 = time.time()
            results = model.track(frame_enhanced, persist=True, conf=TRACK_CONF, iou=TRACK_IOU)
            t1 = time.time()
            # record inference time
            inference_stats_by_file[filename]["total_time"] += (t1 - t0)
            inference_stats_by_file[filename]["count"] += 1

            if not results or results[0].boxes is None:
                print(f"[frame {count}] no detections")
            else:
                boxes_obj = results[0].boxes

                # safe attribute extraction
                boxes = (
                    boxes_obj.xyxy.int().cpu().tolist()
                    if getattr(boxes_obj, "xyxy", None) is not None
                    else []
                )
                class_ids = (
                    boxes_obj.cls.int().cpu().tolist()
                    if getattr(boxes_obj, "cls", None) is not None
                    else []
                )
                confidences = (
                    boxes_obj.conf.cpu().tolist()
                    if getattr(boxes_obj, "conf", None) is not None
                    else []
                )
                track_ids = (
                    boxes_obj.id.int().cpu().tolist()
                    if getattr(boxes_obj, "id", None) is not None
                    else [None] * len(class_ids)
                )

                # Parse detections
                debug_list = []
                for i, cid in enumerate(class_ids):
                    conf = confidences[i] if i < len(confidences) else 0.0
                    label = names[cid] if isinstance(names, (list, dict)) else str(cid)
                    debug_list.append((label, float(conf)))
                print(f"[frame {count}] detections:", debug_list)

                # draw all boxes (accident red, others green) and save crops + frame
                for i in range(len(class_ids)):
                    box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
                    class_id = class_ids[i]
                    track_id = track_ids[i] if i < len(track_ids) else None
                    conf = confidences[i] if i < len(confidences) else 0.0

                    # Get raw label and apply remapping
                    label_raw = names[class_id] if isinstance(names, (list, dict)) else str(class_id)
                    label = LABEL_REMAP.get(label_raw, LABEL_REMAP.get(label_raw.lower(), label_raw))

                    x1, y1, x2, y2 = map(int, box)
                    is_accident_high_conf = (label.lower() == ACCIDENT_CLASS_NAME.lower() and conf >= ACCIDENT_CONF_THRESHOLD)
                    is_accident_low_conf = (label.lower() == ACCIDENT_CLASS_NAME.lower() and conf < ACCIDENT_CONF_THRESHOLD)
                    is_vehicle_class = label.lower() in [v.lower() for v in TARGET_VEHICLE_CLASSES]
                    
                    # Skip drawing for:
                    # 1. Low-confidence accidents
                    if is_accident_low_conf:
                        continue
                    
                    # Show all other detections as green boxes (including non-vehicle classes)
                    if is_accident_high_conf:
                        color = (0, 0, 255)  # Red for high-confidence accidents
                    elif is_vehicle_class:
                        color = (255, 0, 0)  # Blue for vehicle classes
                    else:
                        color = (0, 255, 0)  # Green for other classes

                    # draw rectangle and text (accident high-conf or vehicle classes)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{track_id} - {label} ({conf:.2f})",
                        (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                    frame_has_detection = True

                    # record detected object label
                    detected_objects.append(label)

                    # เก็บทุกรายการ detection เพื่อใช้ในการ evaluate แบบสี
                    detections.append({
                        "frame": count,
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "label": label,
                        "is_accident": bool(is_accident_high_conf)
                    })

                    # if it's a confident accident, save full-frame (with drawn boxes) and log it
                    if is_accident_high_conf:
                        frame_filename = f"{filename}_frame_{count}.jpg"
                        frame_path = os.path.join(ACCIDENT_FRAME_DIR, frame_filename)
                        try:
                            cv2.imwrite(frame_path, frame)  # saved AFTER drawing boxes
                        except Exception as _e:
                            print(f"Failed to write accident frame {frame_path}: {_e}")

                        sec = int(count / fps) if fps else count
                        if not accident_log or accident_log[-1].get("frame") != count:
                            accident_log.append(
                                {
                                    "frame": count,
                                    "time": sec,
                                    "box": [x1, y1, x2, y2],
                                    "confidence": conf,
                                    "img": frame_filename,
                                    "label": label,
                                    "vehicle_counts": {}  # Will be filled after vehicle detection
                                }
                            )
                            print("RED ACCIDENT LOGGED:", frame_filename, conf)
                            # Store accident info for this frame for vehicle detection
                            found_accident_this_frame = {
                                "box": [x1, y1, x2, y2],
                                "conf": conf,
                                "label": label,
                                "log_index": len(accident_log) - 1  # Index to update with vehicle counts
                            }

            # --- overlay summary (total boxes & per-class counts) and save full-frame detection ---
            total_boxes = len(class_ids)
            class_labels = [names[cid] if isinstance(names, (list, dict)) else str(cid) for cid in class_ids]
            counts = Counter(class_labels)

            # Initialize vehicle counts
            vehicle_counts = {}

            # --- If accident found, detect vehicles in accident ROI ---
            if found_accident_this_frame and vehicle_model:
                x1, y1, x2, y2 = found_accident_this_frame["box"]
                roi_frame = frame_enhanced[y1:y2, x1:x2]  # Use enhanced frame

                if roi_frame.size > 0:
                    try:
                        results_veh = vehicle_model.predict(roi_frame, verbose=False, conf=VEHICLE_DETECTION_CONF)
                        for res_v in results_veh:
                            boxes_v = res_v.boxes
                            for box_v in boxes_v:
                                v_cls_id = int(box_v.cls[0])
                                v_cls_name = vehicle_model.names[v_cls_id] if isinstance(vehicle_model.names, (list, dict)) else str(v_cls_id)

                                if v_cls_name.lower() in [v.lower() for v in TARGET_VEHICLE_CLASSES]:
                                    vx1_local, vy1_local, vx2_local, vy2_local = map(int, box_v.xyxy[0])

                                    if v_cls_name in vehicle_counts:
                                        vehicle_counts[v_cls_name] += 1
                                    else:
                                        vehicle_counts[v_cls_name] = 1

                                    vx1 = vx1_local + x1
                                    vy1 = vy1_local + y1
                                    vx2 = vx2_local + x1
                                    vy2 = vy2_local + y1

                                    # Draw vehicle box on frame (green)
                                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                                    cv2.putText(frame, v_cls_name, (vx1, vy1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        print(f"   -> Vehicle summary: {vehicle_counts}")
                        
                        # Update accident log with vehicle counts
                        if "log_index" in found_accident_this_frame:
                            log_idx = found_accident_this_frame["log_index"]
                            if 0 <= log_idx < len(accident_log):
                                accident_log[log_idx]["vehicle_counts"] = vehicle_counts
                    except Exception as e:
                        print(f"   -> Error detecting vehicles: {e}")
            
            # --- Optional: Full-frame vehicle detection (disabled by default for performance) ---
            # Uncomment to enable, but will add latency
            # if DETECT_VEHICLES_FULL_FRAME and vehicle_model:
            #     try:
            #         results_full = vehicle_model.predict(frame_enhanced, verbose=False, conf=0.5)
            #         for res_f in results_full:
            #             boxes_f = res_f.boxes
            #             for box_f in boxes_f:
            #                 f_cls_id = int(box_f.cls[0])
            #                 f_cls_name = vehicle_model.names[f_cls_id] if isinstance(vehicle_model.names, (list, dict)) else str(f_cls_id)
            #                 if f_cls_name.lower() in [v.lower() for v in TARGET_VEHICLE_CLASSES]:
            #                     fx1, fy1, fx2, fy2 = map(int, box_f.xyxy[0])
            #                     is_in_accident_area = False
            #                     if found_accident_this_frame:
            #                         ax1, ay1, ax2, ay2 = found_accident_this_frame["box"]
            #                         if not (fx2 < ax1 or fx1 > ax2 or fy2 < ay1 or fy1 > ay2):
            #                             is_in_accident_area = True
            #                     if not is_in_accident_area:
            #                         cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
            #                         cv2.putText(frame, f"{f_cls_name}", (fx1, fy1 - 5),
            #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #     except Exception as e:
            #         pass
            
            # --- Send Telegram alert in background (non-blocking) ---
            if found_accident_this_frame:
                import threading
                alert_thread = threading.Thread(
                    target=send_telegram_alert_async,
                    args=(frame, found_accident_this_frame["label"], 
                          found_accident_this_frame["conf"] * 100, vehicle_counts, filename),
                    daemon=True
                )
                alert_thread.start()

            overlay_text = f"Detections: {total_boxes}"
            cv2.putText(frame, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            y = 45
            for lbl, cnt in counts.items():
                line = f"{lbl}: {cnt}"
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y += 20

            # save full-frame detection image only when accident is detected
            if found_accident_this_frame:
                det_frame_filename = f"{filename}_detframe_{count}.jpg"
                det_frame_path = os.path.join(DETECTION_FRAME_DIR, det_frame_filename)
                try:
                    cv2.imwrite(det_frame_path, frame)
                except Exception as _e:
                    print(f"Failed to write detection frame {det_frame_path}: {_e}")

        except Exception as e:
            print(f"Exception processing frame {count}:", e)

        # update shared state for frontend
        detected_objects_by_file[filename] = detected_objects
        accident_log_by_file[filename] = accident_log
        all_detections_by_file[filename] = detections

        # encode frame for MJPEG stream (frame with boxes)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/video_feed/<filename>")
def video_feed(filename):
    video_path = os.path.join("uploads", filename)
    return Response(detect_objects_from_video(video_path, filename), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/uploads/<filename>")
def play_video(filename):
    detected_objects = detected_objects_by_file.get(filename, [])
    return render_template("play_video.html", filename=filename, detected_objects=detected_objects)


@app.route("/video/<path:filename>")
def send_video(filename):
    return send_from_directory("uploads", filename)


@app.route("/accident_log/<filename>")
def accident_log(filename):
    log = accident_log_by_file.get(filename, [])
    return jsonify(log)



@app.route("/detection_frames/<filename>")
def get_detection_frames(filename):
    """Return list of all detection frame numbers for a video"""
    frames = []
    if filename in all_detections_by_file:
        # Get unique frame numbers from detections
        frame_nums = set()
        for detection in all_detections_by_file[filename]:
            frame_nums.add(detection['frame'])
        frames = sorted(list(frame_nums))
    
    # If no detections, try to find saved detection frames
    if not frames:
        pattern = os.path.join(DETECTION_FRAME_DIR, f"{filename}_detframe_*.jpg")
        frame_files = glob.glob(pattern)
        frames = []
        for f in frame_files:
            basename = os.path.basename(f)
            # Extract frame number from filename like "video_detframe_123.jpg"
            try:
                frame_num = int(basename.split('_detframe_')[1].split('.')[0])
                frames.append(frame_num)
            except:
                continue
        frames = sorted(list(set(frames)))
    
    return jsonify(frames)


@app.route("/detection_frame/<filename>")
def detection_frame(filename):
    frame_num = int(request.args.get("frame", 0))
    detection_frame_filename = f"{filename}_detframe_{frame_num}.jpg"
    frame_path = os.path.join(DETECTION_FRAME_DIR, detection_frame_filename)
    if not os.path.exists(frame_path):
        return "", 404
    return send_file(frame_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)