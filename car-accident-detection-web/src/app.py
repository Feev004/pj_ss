import csv
import datetime
import glob
import hashlib
import io
import ipaddress
import os
import socket
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import pytz
import requests
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_cors import CORS
from PIL import Image as PILImage
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Enable CORS for Next.js frontend communication
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Limit maximum upload size to 500 MB (DoS protection)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "accident_detection_secret_key_2026")

# --- DIRECTORY CONFIGURATION (Absolute paths to avoid CWD ambiguity) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ACCIDENT_FRAME_DIR = os.path.join(BASE_DIR, "accident_frames")
DETECTION_FRAME_DIR = os.path.join(BASE_DIR, "detection_frames")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ACCIDENT_FRAME_DIR, exist_ok=True)
os.makedirs(DETECTION_FRAME_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_DOWNLOAD_SIZE = 250 * 1024 * 1024  # 250 MB limit for URL downloads

# --- SECURITY & THREAD LOCK ---
data_lock = threading.Lock()

def safe_filename(name):
    """Sanitize filename to prevent directory traversal attacks"""
    if not name:
        return ""
    clean = secure_filename(os.path.basename(name))
    return clean

def is_allowed_file(filename):
    """Validate file extension against whitelist"""
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS

def is_safe_url(url_str):
    """Prevent SSRF attacks against internal network, localhost, or cloud metadata endpoints"""
    try:
        parsed = urlparse(url_str)
        if parsed.scheme not in ("http", "https"):
            return False, "โปรโตคอลต้องเป็น http หรือ https เท่านั้น"
        hostname = parsed.hostname
        if not hostname:
            return False, "ไม่พบ hostname ใน URL"
        if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
            return False, "ไม่อนุญาตให้เข้าถึง localhost หรือที่อยู่ภายในระบบ"
        
        # Resolve hostname to IP address
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, "ไม่อนุญาตให้เข้าถึงที่อยู่ IP เครือข่ายภายใน (SSRF Protection)"
        return True, ""
    except Exception as e:
        return False, f"ไม่สามารถตรวจสอบความปลอดภัยของ URL ได้: {e!s}"

# --- CONFIGURATION ---
model_path = os.path.join(BASE_DIR, "model", "best.pt")
model = YOLO(model_path)
names = getattr(model, "names", {})
if not names and getattr(model, "model", None) is not None:
    names = getattr(model.model, "names", {})
print("MODEL CLASS NAMES:", names)

# Telegram Configuration (supports environment variables)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7692550301:AAGPTJrcCDdXuitHRNQep85UhfSUcXMxNkc")
TARGET_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-4887423941")
ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "False").lower() in ("true", "1", "yes")

# Vehicle detection model (สำหรับนับรถในกรอบ accident)
VEHICLE_MODEL_PATHS = [
    os.path.join(BASE_DIR, "..", "..", "testmo", "model11n", "yolov11n.pt"),
    os.path.join(BASE_DIR, "..", "testmo", "model11n", "yolov11n.pt"),
    os.path.join(BASE_DIR, "model", "yolov11n.pt"),
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
    print("⚠️  ไม่พบไฟล์โมเดล vehicle - จะใช้เฉพาะ accident detection")

TARGET_VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

# ===== PERFORMANCE & INTELLIGENT ACCIDENT DETECTION SETTINGS =====
ACCIDENT_CLASS_NAME = "accident"
ACCIDENT_CONF_THRESHOLD = 0.70  # Sensitivity threshold for accident candidates
PROCESS_EVERY_N = 2             # Frame skip (smooth 15fps playback from 30fps)
TRACK_CONF = 0.50               # Tracker confidence
TRACK_IOU = 0.45
VEHICLE_DETECTION_CONF = 0.40
DETECT_VEHICLES_FULL_FRAME = False

# --- SMART TEMPORAL & SPATIAL FILTERS (False Positive Reducer) ---
MIN_ACCIDENT_STREAK = 2          # Must be detected across at least 2 consecutive processed frames
ACCIDENT_COOLDOWN_SEC = 3.5      # Minimum seconds between distinct logged accident events (prevents log flooding)
CONFIRMED_HIGH_CONF = 0.85       # If single frame confidence is >= 85%, mark confirmed immediately

LABEL_REMAP = {
    "non-accident": "car",
    "Non-Accident": "car",
    "non_accident": "car",
}

detected_objects_by_file = {}
accident_log_by_file = {}
all_detections_by_file = {}   # Store all detections for evaluation
inference_stats_by_file = {}  # Track inference time per file

# --- HELPER FUNCTIONS ---
def apply_adaptive_enhancement(image):
    """
    Apply Adaptive Contrast & Lighting Enhancement:
    Automatically measures average luminosity to adaptively enhance contrast
    without washing out daylight frames or leaving night scenes in pitch dark.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_lum = float(np.mean(l))

    # Low-light / Night enhancement
    if mean_lum < 75.0:
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    elif mean_lum > 185.0:
        # High-glare / Bright daylight
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    else:
        # Normal lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_img

# Alias for backward compatibility
apply_clahe = apply_adaptive_enhancement


def to_list_safe(val, to_int=False):
    """Safely converts PyTorch tensor, numpy array, or list to Python list without type errors."""
    if val is None:
        return []
    try:
        if hasattr(val, "cpu"):
            val = val.cpu()
        if hasattr(val, "numpy"):
            val = val.numpy()
        if isinstance(val, np.ndarray):
            if to_int:
                val = val.astype(int)
            return val.tolist()
        if hasattr(val, "tolist"):
            return val.tolist()
        return list(val)
    except Exception:
        return []


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
        print("✅ ส่งแจ้งเตือน Telegram สำเร็จ!")
    except Exception as e:
        print(f"❌ Error sending telegram: {e}")


def send_telegram_alert_async(image_array, detected_class, conf, vehicle_counts, filename):
    """Send alert notification to Telegram (runs in background thread)"""
    if not ENABLE_TELEGRAM or not TARGET_CHAT_ID:
        return
    try:
        send_telegram_alert(image_array, detected_class, conf, vehicle_counts, filename)
    except Exception as e:
        print(f"⚠️  Telegram alert error: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    is_json = (
        request.headers.get("Accept") == "application/json"
        or request.is_json
        or request.args.get("format") == "json"
    )

    if "file" not in request.files:
        err = "ไม่พบไฟล์ที่อัปโหลด"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(request.url)

    file = request.files["file"]
    if not file or file.filename == "":
        err = "ไม่ได้เลือกไฟล์"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(request.url)

    if not is_allowed_file(file.filename):
        err = f"นามสกุลไฟล์ไม่ถูกต้อง รองรับเฉพาะ: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(url_for("index"))

    filename = safe_filename(file.filename)
    if not filename:
        err = "ชื่อไฟล์ไม่ถูกต้อง"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(url_for("index"))

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # ตรวจสอบความถูกต้องของวิดีโอด้วย OpenCV
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        cap.release()
        if os.path.exists(file_path):
            os.remove(file_path)
        err = "ไฟล์ที่อัปโหลดไม่สามารถเปิดเป็นวิดีโอที่ถูกต้องได้"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(url_for("index"))
    cap.release()

    # --- สร้างไฟล์ GT อัตโนมัติถ้ายังไม่มี หรือถ้าว่าง ให้เติมตัวอย่าง ---
    base = os.path.splitext(filename)[0]
    gt_path = os.path.join(UPLOAD_FOLDER, f"{base}_gt.txt")
    example_path = os.path.join(UPLOAD_FOLDER, f"{base}_gt_example.txt")
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

    with data_lock:
        detected_objects_by_file[filename] = []
        accident_log_by_file[filename] = []
        all_detections_by_file[filename] = []
        inference_stats_by_file[filename] = {"total_time": 0.0, "count": 0}

    if is_json:
        return jsonify({
            "success": True,
            "filename": filename,
            "video_url": f"/video/{filename}",
            "feed_url": f"/video_feed/{filename}",
            "redirect_url": url_for("play_video", filename=filename),
        })

    return redirect(url_for("play_video", filename=filename))


@app.route("/upload_url", methods=["POST"])
def upload_video_from_url():
    """Upload video from URL with SSRF protection and download capping"""
    is_json = (
        request.headers.get("Accept") == "application/json"
        or request.is_json
        or request.args.get("format") == "json"
    )

    req_data = request.get_json(silent=True) or request.form
    video_url = req_data.get("video_url", "").strip()

    if not video_url:
        err = "กรุณาป้อน URL วิดีโอ"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(url_for("index"))

    # SSRF Protection Check
    is_safe, err_msg = is_safe_url(video_url)
    if not is_safe:
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        flash(err_msg, "error")
        return redirect(url_for("index"))

    try:
        url_hash = hashlib.md5(video_url.encode()).hexdigest()[:8]
        filename = f"url_{url_hash}.mp4"
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AccidentDetection/1.0"}
        response = requests.get(video_url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()

        # Check content-length header
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            raise ValueError(f"ขนาดไฟล์ ({int(content_length)/(1024*1024):.1f} MB) เกินขีดจำกัดสูงสุด {MAX_DOWNLOAD_SIZE/(1024*1024)} MB")

        downloaded = 0
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=16384):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_SIZE:
                        break
                    f.write(chunk)

        if downloaded > MAX_DOWNLOAD_SIZE:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise ValueError(f"ขนาดไฟล์ดาวน์โหลดเกินขีดจำกัดสูงสุด {MAX_DOWNLOAD_SIZE/(1024*1024)} MB")

        # Verify video with OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            cap.release()
            if os.path.exists(file_path):
                os.remove(file_path)
            err = "ไฟล์ที่ดาวน์โหลดไม่ใช่วิดีโอที่ถูกต้อง"
            if is_json:
                return jsonify({"success": False, "error": err}), 400
            flash(err, "error")
            return redirect(url_for("index"))
        cap.release()

        # Create GT file
        base = os.path.splitext(filename)[0]
        gt_path = os.path.join(UPLOAD_FOLDER, f"{base}_gt.txt")
        example_lines = [
            "# GT format per line: frame,x1,y1,x2,y2,label",
            "# ตัวอย่าง:",
            "15,100,50,420,300,accident",
        ]

        if not os.path.exists(gt_path) or os.path.getsize(gt_path) == 0:
            with open(gt_path, "w", encoding="utf-8") as gf:
                gf.write("\n".join(example_lines) + "\n")

        with data_lock:
            detected_objects_by_file[filename] = []
            accident_log_by_file[filename] = []
            all_detections_by_file[filename] = []
            inference_stats_by_file[filename] = {"total_time": 0.0, "count": 0}

        if is_json:
            return jsonify({
                "success": True,
                "filename": filename,
                "video_url": f"/video/{filename}",
                "feed_url": f"/video_feed/{filename}",
                "redirect_url": url_for("play_video", filename=filename),
            })

        flash(f"ดาวน์โหลดวิดีโอจาก URL สำเร็จ: {filename}", "success")
        return redirect(url_for("play_video", filename=filename))

    except Exception as e:
        err = f"เกิดข้อผิดพลาดในการประมวลผล URL: {e!s}"
        if is_json:
            return jsonify({"success": False, "error": err}), 400
        flash(err, "error")
        return redirect(url_for("index"))


@app.route("/detected_objects/<filename>")
def get_detected_objects(filename):
    clean_name = safe_filename(filename)
    with data_lock:
        detected_objects = list(detected_objects_by_file.get(clean_name, []))
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
    clean_name = safe_filename(filename)
    base = os.path.splitext(clean_name)[0]
    candidates = [
        os.path.join(UPLOAD_FOLDER, f"{base}_gt.txt"),
        os.path.join(UPLOAD_FOLDER, f"{base}_gt.csv"),
        os.path.join(UPLOAD_FOLDER, f"{base}.gt"),
        os.path.join(UPLOAD_FOLDER, f"{base}.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            gt = []
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    line = ",".join(row).strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = [s.strip() for s in line.split(",")]
                    if len(parts) < 5:
                        continue
                    try:
                        frame = int(parts[0])
                        x1, y1, x2, y2 = map(int, parts[1:5])
                    except Exception:
                        continue
                    label = parts[5].strip() if len(parts) > 5 else "accident"
                    label = LABEL_REMAP.get(label, LABEL_REMAP.get(label.lower(), label))
                    gt.append({"frame": frame, "box": [x1, y1, x2, y2], "label": label})
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
        mpre[i] = max(mpre[i], mpre[i + 1])
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
    clean_name = safe_filename(filename)
    try:
        iou_thr = float(request.args.get("iou", 0.5))
    except Exception:
        iou_thr = 0.5

    gts = load_gt_for_video(clean_name) or []
    with data_lock:
        preds = list(all_detections_by_file.get(clean_name, []) or [])
        stats = dict(inference_stats_by_file.get(clean_name, {"total_time": 0.0, "count": 0}))

    # existing color-count style metrics (kept for backward compatibility)
    total_gt = sum(1 for _ in gts) if gts is not None else 0
    dets = preds
    total_pred = len(dets)

    # นับ TP +1 สำหรับทุกการตรวจจับ (ทุกค่า) และนับ FP +1 เมื่อกรอบเป็นสีเขียว (non-accident)
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
    stats = inference_stats_by_file.get(clean_name, {"total_time": 0.0, "count": 0})
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

    # --- INTELLIGENT FILTER STATE ---
    accident_streak = 0
    last_logged_accident_time = -999.0

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

        # Apply Adaptive Lighting Enhancement (Handles night / bright glare automatically)
        frame_enhanced = apply_adaptive_enhancement(frame)

        # initialize per-frame variables
        class_ids = []
        boxes = []
        confidences = []
        track_ids = []
        detected_objects = []
        raw_accident_candidate = None

        try:
            t0 = time.time()
            results = model.track(frame_enhanced, persist=True, conf=TRACK_CONF, iou=TRACK_IOU)
            t1 = time.time()
            # record inference time
            inference_stats_by_file[filename]["total_time"] += (t1 - t0)
            inference_stats_by_file[filename]["count"] += 1

            if results and results[0].boxes is not None:
                boxes_obj = results[0].boxes
                boxes = to_list_safe(getattr(boxes_obj, "xyxy", None), to_int=True)
                class_ids = to_list_safe(getattr(boxes_obj, "cls", None), to_int=True)
                confidences = to_list_safe(getattr(boxes_obj, "conf", None), to_int=False)
                track_ids_list = to_list_safe(getattr(boxes_obj, "id", None), to_int=True)
                track_ids = track_ids_list if track_ids_list else [None] * len(class_ids)

                # Process all detected boxes
                for i in range(len(class_ids)):
                    box = boxes[i] if i < len(boxes) else [0, 0, 0, 0]
                    class_id = class_ids[i]
                    track_id = track_ids[i] if i < len(track_ids) else None
                    conf = confidences[i] if i < len(confidences) else 0.0

                    label_raw = names[class_id] if isinstance(names, (list, dict)) else str(class_id)
                    label = LABEL_REMAP.get(label_raw, LABEL_REMAP.get(label_raw.lower(), label_raw))

                    x1, y1, x2, y2 = map(int, box)
                    is_accident_cand = (
                        label.lower() == ACCIDENT_CLASS_NAME.lower()
                        and conf >= ACCIDENT_CONF_THRESHOLD
                    )
                    is_vehicle = label.lower() in [v.lower() for v in TARGET_VEHICLE_CLASSES]

                    if is_accident_cand:
                        if (
                            raw_accident_candidate is None
                            or conf > raw_accident_candidate["conf"]
                        ):
                            raw_accident_candidate = {
                                "box": [x1, y1, x2, y2],
                                "conf": conf,
                                "label": label,
                                "track_id": track_id,
                            }
                    elif is_vehicle:
                        color = (255, 0, 0)  # Blue for vehicles
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"{label} ({conf:.2f})",
                            (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )
                        detected_objects.append(label)
                        detections.append({
                            "frame": count,
                            "box": [x1, y1, x2, y2],
                            "confidence": conf,
                            "label": label,
                            "is_accident": False,
                        })
                    else:
                        color = (0, 255, 0)  # Green for other objects
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"{label} ({conf:.2f})",
                            (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                        )
                        detected_objects.append(label)

            # --- VEHICLE CROSS-VALIDATION & ROI REFINEMENT ---
            vehicle_counts = {}
            is_spatial_verified = False

            if raw_accident_candidate:
                ax1, ay1, ax2, ay2 = raw_accident_candidate["box"]
                h_b, w_b = ay2 - ay1, ax2 - ax1
                pad_x, pad_y = int(w_b * 0.15), int(h_b * 0.15)
                rx1, ry1 = max(0, ax1 - pad_x), max(0, ay1 - pad_y)
                rx2, ry2 = min(frame_enhanced.shape[1], ax2 + pad_x), min(
                    frame_enhanced.shape[0], ay2 + pad_y
                )

                if vehicle_model:
                    roi_frame = frame_enhanced[ry1:ry2, rx1:rx2]
                    if roi_frame.size > 0:
                        try:
                            results_veh = vehicle_model.predict(
                                roi_frame, verbose=False, conf=VEHICLE_DETECTION_CONF
                            )
                            veh_names = getattr(vehicle_model, "names", {})
                            for res_v in results_veh:
                                if res_v.boxes is not None:
                                    for box_v in res_v.boxes:
                                        cls_data = getattr(box_v, "cls", None)
                                        if cls_data is None or len(cls_data) == 0:
                                            continue
                                        v_cls_id = int(cls_data[0])
                                        v_cls_name = (
                                            veh_names.get(v_cls_id, str(v_cls_id))
                                            if isinstance(veh_names, dict)
                                            else (
                                                veh_names[v_cls_id]
                                                if isinstance(veh_names, list) and v_cls_id < len(veh_names)
                                                else str(v_cls_id)
                                            )
                                        )
                                        if v_cls_name.lower() in [
                                            v.lower() for v in TARGET_VEHICLE_CLASSES
                                        ]:
                                            vehicle_counts[v_cls_name] = (
                                                vehicle_counts.get(v_cls_name, 0) + 1
                                            )
                                            xyxy_data = getattr(box_v, "xyxy", None)
                                            if xyxy_data is None or len(xyxy_data) == 0:
                                                continue
                                            vx1_loc, vy1_loc, vx2_loc, vy2_loc = map(
                                                int, xyxy_data[0]
                                            )
                                            vx1, vy1 = vx1_loc + rx1, vy1_loc + ry1
                                            vx2, vy2 = vx2_loc + rx1, vy2_loc + ry1
                                            # Highlight vehicles in accident area (Bright Green)
                                            cv2.rectangle(
                                                frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2
                                            )
                                            cv2.putText(
                                                frame,
                                                f"{v_cls_name}",
                                                (vx1, max(15, vy1 - 5)),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.45,
                                                (0, 255, 0),
                                                1,
                                            )

                            total_roi_vehs = sum(vehicle_counts.values())
                            # Spatial logic: collision of 2+ vehicles, or single vehicle with high conf, or very high confidence
                            if total_roi_vehs >= 2 or total_roi_vehs == 1 and raw_accident_candidate["conf"] >= 0.72 or raw_accident_candidate["conf"] >= CONFIRMED_HIGH_CONF:
                                is_spatial_verified = True
                            else:
                                is_spatial_verified = False
                        except Exception:
                            is_spatial_verified = True
                else:
                    is_spatial_verified = True

            # --- TEMPORAL PERSISTENCE FILTER (Filters single-frame flickers) ---
            if raw_accident_candidate and is_spatial_verified:
                accident_streak += 1
            else:
                accident_streak = max(0, accident_streak - 1)

            is_confirmed_accident = accident_streak >= MIN_ACCIDENT_STREAK

            # --- VISUAL FEEDBACK & OVERLAY ---
            if raw_accident_candidate:
                ax1, ay1, ax2, ay2 = raw_accident_candidate["box"]
                a_conf = raw_accident_candidate["conf"]
                detected_objects.append("accident")

                if is_confirmed_accident:
                    # CONFIRMED: Prominent Red Box with Badge
                    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 0, 255), 3)
                    label_text = f"🚨 ACCIDENT CONFIRMED ({a_conf * 100:.1f}%)"
                    (tw, _), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame, (ax1, max(0, ay1 - 25)), (ax1 + tw + 10, ay1), (0, 0, 255), -1
                    )
                    cv2.putText(
                        frame,
                        label_text,
                        (ax1 + 5, max(15, ay1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    detections.append({
                        "frame": count,
                        "box": raw_accident_candidate["box"],
                        "confidence": a_conf,
                        "label": ACCIDENT_CLASS_NAME,
                        "is_accident": True,
                    })

                    # --- LOGGING WITH INCIDENT COOLDOWN (Eliminates spam) ---
                    sec = int(count / fps) if fps else count
                    time_since_last_alert = sec - last_logged_accident_time
                    if time_since_last_alert >= ACCIDENT_COOLDOWN_SEC:
                        last_logged_accident_time = sec
                        frame_filename = f"{filename}_frame_{count}.jpg"
                        frame_path = os.path.join(ACCIDENT_FRAME_DIR, frame_filename)
                        try:
                            cv2.imwrite(frame_path, frame)
                        except Exception as _e:
                            pass

                        det_frame_filename = f"{filename}_detframe_{count}.jpg"
                        det_frame_path = os.path.join(
                            DETECTION_FRAME_DIR, det_frame_filename
                        )
                        try:
                            cv2.imwrite(det_frame_path, frame)
                        except Exception as _e:
                            pass

                        accident_log.append({
                            "frame": count,
                            "time": sec,
                            "box": raw_accident_candidate["box"],
                            "confidence": a_conf,
                            "img": frame_filename,
                            "label": raw_accident_candidate["label"],
                            "vehicle_counts": vehicle_counts,
                        })
                        print(
                            f"🚨 [CONFIRMED ACCIDENT] Frame {count} ({sec}s) | Conf: {a_conf:.2f} | Vehicles: {vehicle_counts}"
                        )

                        if ENABLE_TELEGRAM:
                            alert_thread = threading.Thread(
                                target=send_telegram_alert_async,
                                args=(
                                    frame,
                                    raw_accident_candidate["label"],
                                    a_conf * 100,
                                    vehicle_counts,
                                    filename,
                                ),
                                daemon=True,
                            )
                            alert_thread.start()

                elif accident_streak == 1:
                    # CANDIDATE (PENDING CONFIRMATION): Amber box (no spam, no log yet)
                    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 165, 255), 2)
                    label_text = f"Analyzing... ({a_conf * 100:.1f}%)"
                    cv2.putText(
                        frame,
                        label_text,
                        (ax1, max(15, ay1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        1,
                    )

            # Overlay stats HUD on frame
            status_text = (
                f"Status: {'CRITICAL (ACCIDENT)' if is_confirmed_accident else 'MONITORING'} | Streak: {accident_streak}"
            )
            status_color = (0, 0, 255) if is_confirmed_accident else (0, 255, 0)
            cv2.putText(
                frame,
                status_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                status_color,
                2,
                cv2.LINE_AA,
            )

        except Exception as e:
            print(f"Exception processing frame {count}:", e)

        # update shared state for frontend (thread-safe)
        with data_lock:
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
    clean_name = safe_filename(filename)
    if not clean_name:
        return "Invalid filename", 400
    video_path = os.path.join(UPLOAD_FOLDER, clean_name)
    if not os.path.exists(video_path):
        return "Video not found", 404
    return Response(
        detect_objects_from_video(video_path, clean_name),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/uploads/<filename>")
def play_video(filename):
    clean_name = safe_filename(filename)
    with data_lock:
        detected_objects = list(detected_objects_by_file.get(clean_name, []))
    return render_template("play_video.html", filename=clean_name, detected_objects=detected_objects)


@app.route("/video/<path:filename>")
def send_video(filename):
    clean_name = safe_filename(filename)
    if not clean_name:
        return "Invalid filename", 400
    return send_from_directory(UPLOAD_FOLDER, clean_name)


@app.route("/accident_log/<filename>")
def accident_log(filename):
    clean_name = safe_filename(filename)
    with data_lock:
        log = list(accident_log_by_file.get(clean_name, []))
    return jsonify(log)


@app.route("/accident_frame/<filename>")
def get_accident_frame(filename):
    clean_name = safe_filename(filename)
    if not clean_name:
        return "", 404
    return send_from_directory(ACCIDENT_FRAME_DIR, clean_name)


@app.route("/detection_frames/<filename>")
def get_detection_frames(filename):
    """Return list of all detection frame numbers for a video"""
    clean_name = safe_filename(filename)
    frames = []
    with data_lock:
        if clean_name in all_detections_by_file:
            frame_nums = set()
            for detection in all_detections_by_file[clean_name]:
                frame_nums.add(detection["frame"])
            frames = sorted(frame_nums)

    # If no detections in memory, check saved detection frames
    if not frames:
        pattern = os.path.join(DETECTION_FRAME_DIR, f"{clean_name}_detframe_*.jpg")
        frame_files = glob.glob(pattern)
        frames = []
        for f in frame_files:
            basename = os.path.basename(f)
            try:
                frame_num = int(basename.split("_detframe_")[1].split(".")[0])
                frames.append(frame_num)
            except Exception:
                continue
        frames = sorted(set(frames))

    return jsonify(frames)


@app.route("/detection_frame/<filename>")
def detection_frame(filename):
    clean_name = safe_filename(filename)
    if not clean_name:
        return "", 404
    try:
        frame_num = int(request.args.get("frame", 0))
    except (ValueError, TypeError):
        return "", 400
    detection_frame_filename = f"{clean_name}_detframe_{frame_num}.jpg"
    return send_from_directory(DETECTION_FRAME_DIR, detection_frame_filename)


@app.route("/api/videos", methods=["GET"])
def list_videos():
    """List available uploaded videos for the dashboard"""
    videos = []
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            f_path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(f_path) and is_allowed_file(f):
                stat = os.stat(f_path)
                videos.append({
                    "filename": f,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
    videos.sort(key=lambda x: x["created_at"], reverse=True)
    return jsonify({"success": True, "videos": videos})


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """System health and AI configuration endpoint"""
    return jsonify({
        "status": "online",
        "models": {
            "accident_model": model is not None,
            "vehicle_model": vehicle_model is not None,
        },
        "config": {
            "enable_telegram": ENABLE_TELEGRAM,
            "accident_conf_threshold": ACCIDENT_CONF_THRESHOLD,
            "track_conf": TRACK_CONF,
            "process_every_n": PROCESS_EVERY_N,
            "clahe_enabled": True
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)