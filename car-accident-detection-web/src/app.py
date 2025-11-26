import csv
import math
import os
from collections import Counter
import time

import cv2
from flask import (
    Flask,
    Response,
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

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(model_path)
names = model.model.names
print("MODEL CLASS NAMES:", names)

# thresholds (ปรับเพื่อทดสอบ)
ACCIDENT_CLASS_NAME = "accident"  # เปลี่ยนถ้าชื่อคลาสใน model ต่างกัน
ACCIDENT_CONF_THRESHOLD = 0.1
PROCESS_EVERY_N = 1
TRACK_CONF = 0.45  # ค่าเริ่มต้นสำหรับ model.track ตอนดีบัก -> ปรับเพิ่มเมื่อพร้อม
TRACK_IOU = 0.45

LABEL_REMAP = {
    "non-accident": "car",
    "Non-Accident": "car",
    "non_accident": "car",
}

detected_objects_by_file = {}
accident_log_by_file = {}
all_detections_by_file = {}   # <-- เพิ่มตัวเก็บทุก detection (ทุกคลาส / ทุกเฟรม)
inference_stats_by_file = {}  # เก็บเวลาการทำ inference: {'total_time': float_seconds, 'count': int}

ACCIDENT_FRAME_DIR = "accident_frames"
os.makedirs(ACCIDENT_FRAME_DIR, exist_ok=True)

DETECTION_FRAME_DIR = "detection_frames"
os.makedirs(DETECTION_FRAME_DIR, exist_ok=True)


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
    # filter preds for class and sort by confidence desc
    preds_c = [p for p in preds if p.get("label") == class_name]
    preds_c.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    # index GTs by frame and mark unmatched
    gt_by_frame = {}
    total_gts = 0
    for i, g in enumerate(gts):
        if g.get("label") != class_name:
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
        if (d.get("label") == ACCIDENT_CLASS_NAME) and (d.get("confidence", 0.0) < low_conf_threshold)
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
        has_gt_pos = any((g["frame"] == fr and g["label"] == ACCIDENT_CLASS_NAME) for g in gts)
        has_pred_pos = any((p["frame"] == fr and p.get("label") == ACCIDENT_CLASS_NAME) for p in preds)
        if not has_gt_pos and not has_pred_pos:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- compute AP (mAP for single class 'accident') using IoU matching at iou_thr ---
    ap = compute_ap(preds, gts, iou_thr=iou_thr, class_name=ACCIDENT_CLASS_NAME)
    # print("AP",preds, gts, iou_thr=iou_thr, class_name=ACCIDENT_CLASS_NAME)
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

        # initialize per-frame variables
        frame_has_detection = False
        class_ids = []
        boxes = []
        confidences = []
        track_ids = []
        detected_objects = []

        try:
            # run tracker/detector (adjust conf/iou when tuning)
            t0 = time.time()
            results = model.track(frame, persist=True, conf=TRACK_CONF, iou=TRACK_IOU)
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

                # debug print
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
                    is_accident = (label == ACCIDENT_CLASS_NAME and conf >= ACCIDENT_CONF_THRESHOLD)
                    color = (0, 0, 255) if is_accident else (0, 255, 0)

                    # draw rectangle and text for every detection
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

                    # record detected object label (skip 'non-accident' if desired)
                    detected_objects.append(label)

                    # เก็บทุกรายการ detection เพื่อใช้ในการ evaluate แบบสี
                    detections.append({
                        "frame": count,
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "label": label,
                        "is_accident": bool(is_accident)
                    })

                    # if it's a confident accident, save full-frame (with drawn boxes) and log it
                    if is_accident:
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
                                    "label": label,  # <-- เพิ่มตรงนี้ เพื่อให้ evaluate เทียบคลาสได้ถูกต้อง
                                }
                            )
                            print("RED ACCIDENT LOGGED:", frame_filename, conf)

            # --- overlay summary (total boxes & per-class counts) and save full-frame detection ---
            total_boxes = len(class_ids)
            class_labels = [names[cid] if isinstance(names, (list, dict)) else str(cid) for cid in class_ids]
            counts = Counter(class_labels)

            overlay_text = f"Detections: {total_boxes}"
            cv2.putText(frame, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

            y = 45
            for lbl, cnt in counts.items():
                line = f"{lbl}: {cnt}"
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y += 20

            # save full-frame detection image once per frame (after drawing all boxes)
            if frame_has_detection:
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


@app.route("/accident_frame/<filename>")
def accident_frame(filename):
    frame_num = int(request.args.get("frame", 0))
    frame_filename = f"{filename}_frame_{frame_num}.jpg"
    frame_path = os.path.join(ACCIDENT_FRAME_DIR, frame_filename)
    if not os.path.exists(frame_path):
        return "", 404
    return send_file(frame_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)