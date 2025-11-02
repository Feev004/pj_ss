import os
import cv2
from collections import Counter
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify, send_file
from ultralytics import YOLO

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(model_path)
names = model.model.names
print("MODEL CLASS NAMES:", names)

# thresholds (ปรับเพื่อทดสอบ)
ACCIDENT_CLASS_NAME = "accident"   # เปลี่ยนถ้าชื่อคลาสใน model ต่างกัน
ACCIDENT_CONF_THRESHOLD = 0.1
PROCESS_EVERY_N = 1
TRACK_CONF = 0.45   # ค่าเริ่มต้นสำหรับ model.track ตอนดีบัก -> ปรับเพิ่มเมื่อพร้อม
TRACK_IOU = 0.45

detected_objects_by_file = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    detected_objects_by_file[file.filename] = []

    return redirect(url_for('play_video', filename=file.filename))

@app.route('/detected_objects/<filename>')
def get_detected_objects(filename):
    detected_objects = detected_objects_by_file.get(filename, [])
    return jsonify(detected_objects)

accident_log_by_file = {}
ACCIDENT_FRAME_DIR = "accident_frames"
os.makedirs(ACCIDENT_FRAME_DIR, exist_ok=True)
DETECTION_FRAME_DIR = "detection_frames"
os.makedirs(DETECTION_FRAME_DIR, exist_ok=True)

def detect_objects_from_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    accident_log = []

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

        try:
            # run tracker/detector (adjust conf/iou when tuning)
            results = model.track(frame, persist=True, conf=TRACK_CONF, iou=TRACK_IOU)

            detected_objects = []

            if not results or results[0].boxes is None:
                print(f"[frame {count}] no detections")
            else:
                boxes_obj = results[0].boxes

                # safe attribute extraction
                boxes = boxes_obj.xyxy.int().cpu().tolist() if getattr(boxes_obj, "xyxy", None) is not None else []
                class_ids = boxes_obj.cls.int().cpu().tolist() if getattr(boxes_obj, "cls", None) is not None else []
                confidences = boxes_obj.conf.cpu().tolist() if getattr(boxes_obj, "conf", None) is not None else []
                track_ids = boxes_obj.id.int().cpu().tolist() if getattr(boxes_obj, "id", None) is not None else [None] * len(class_ids)

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
                    label = names[class_id] if isinstance(names, (list, dict)) else str(class_id)

                    x1, y1, x2, y2 = map(int, box)
                    is_accident = (label == ACCIDENT_CLASS_NAME and conf >= ACCIDENT_CONF_THRESHOLD)
                    color = (0, 0, 255) if is_accident else (0, 255, 0)

                    # draw rectangle and text for every detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{track_id} - {label} ({conf:.2f})', (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    frame_has_detection = True

                    # --- บันทึกภาพ crop ของกรอบนี้ (ทุกกรอบ) ---
                    # h, w = frame.shape[:2]
                    # x1c, y1c = max(0, x1), max(0, y1)
                    # x2c, y2c = min(w, x2), min(h, y2)
                    # if x2c > x1c and y2c > y1c:
                    #     crop = frame[y1c:y2c, x1c:x2c]
                    #     safe_label = str(label).replace(" ", "_")
                    #     crop_filename = f"{filename}_frame_{count}_box_{i}_{safe_label}_{int(conf*100)}.jpg"
                    #     crop_path = os.path.join(DETECTION_FRAME_DIR, crop_filename)
                        # try:
                        #     cv2.imwrite(crop_path, crop)
                        # except Exception as _e:
                        #     print(f"Failed to write crop {crop_path}: {_e}")
                    # --------------------------------------------------

                    # record detected object label (skip 'non-accident' if desired)
                    if label != "non-accident":
                        detected_objects.append(label)

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
                            accident_log.append({
                                "frame": count,
                                "time": sec,
                                "box": [x1, y1, x2, y2],
                                "confidence": conf,
                                "img": frame_filename
                            })
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

        # encode frame for MJPEG stream (frame with boxes)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('uploads', filename)
    return Response(detect_objects_from_video(video_path, filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploads/<filename>')
def play_video(filename):
    detected_objects = detected_objects_by_file.get(filename, [])
    return render_template('play_video.html', filename=filename, detected_objects=detected_objects)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)

@app.route('/accident_log/<filename>')
def accident_log(filename):
    log = accident_log_by_file.get(filename, [])
    return jsonify(log)

@app.route('/accident_frame/<filename>')
def accident_frame(filename):
    frame_num = int(request.args.get('frame', 0))
    frame_filename = f"{filename}_frame_{frame_num}.jpg"
    frame_path = os.path.join(ACCIDENT_FRAME_DIR, frame_filename)
    if not os.path.exists(frame_path):
        return '', 404
    return send_file(frame_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)