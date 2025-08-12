import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
from ultralytics import YOLO

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "model", "best.pt") #<-- เปลี่ยนเส้นทางโมเดลตามที่คุณต้องการ
model = YOLO(model_path)
names = model.model.names

detected_objects_by_file = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')

def detect_objects_from_webcam():
    count = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

accident_log_by_file = {}  # เพิ่มตัวแปรเก็บ log

def detect_objects_from_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    accident_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        detected_objects = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                c = names[class_id]
                detected_objects.append(c)
                x1, y1, x2, y2 = box
                color = (0, 0, 255) if c == "accident" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                # ถ้าเจอ accident ให้บันทึก log
                if c == "accident":
                    sec = int(count / fps)
                    accident_log.append({
                        "frame": count,
                        "time": sec,
                        "box": [x1, y1, x2, y2]
                        })
                    print("accident detected!", accident_log)

        detected_objects_by_file[filename] = detected_objects
        accident_log_by_file[filename] = accident_log  # อัพเดต log ทุกเฟรม

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
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
    video_path = os.path.join('uploads', filename)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return '', 404
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
