"""
=============================================================================
SafeVision AI - Video Dataset Extraction & Model Fine-Tuning Pipeline
=============================================================================
สคริปต์สำหรับการเตรียมข้อมูลจากคลิปวิดีโอใหม่ และการเทรน/Fine-tune โมเดล YOLO

ฟังก์ชันการทำงาน:
1. extract_frames_from_video: ดึงภาพเฟรมจากคลิปวิดีโอเพื่อเตรียมทำ Labeling
2. create_dataset_structure: สร้างโฟลเดอร์สำหรับ Train / Val ตามมาตรฐาน YOLO
3. train_accident_model: สั่ง Train หรือ Fine-tune โมเดลด้วย YOLO11 / YOLOv8
=============================================================================
"""

import os
import sys
import cv2
from pathlib import Path

# Fix Windows console UTF-8 output encoding
try:
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
except Exception:
    pass


def extract_frames_from_video(video_path: str, output_dir: str, sample_interval_sec: float = 1.0):
    """
    ดึงภาพเฟรมจากคลิปวิดีโอตามช่วงเวลาที่กำหนด เพื่อนำไปตีกรอบ (Annotation)
    
    :param video_path: ที่อยู่ของไฟล์วิดีโอ เช่น 'uploads/new_video.mp4'
    :param output_dir: โฟลเดอร์ปลายทางสำหรับบันทึกรูปภาพ
    :param sample_interval_sec: ดึงภาพทุกๆ กี่วินาที (ค่าเริ่มต้น 1.0 วินาที)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ ไม่สามารถเปิดวิดีโอ: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps * sample_interval_sec))
    base_name = Path(video_path).stem

    count = 0
    saved = 0
    print(f"🎬 กำลังดึงภาพจาก {video_path} (ทุกๆ {sample_interval_sec} วินาที)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_interval == 0:
            out_filename = f"{base_name}_frame_{count}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, frame)
            saved += 1
            
        count += 1

    cap.release()
    print(f"✅ ดึงภาพเสร็จสิ้น! บันทึกภาพทั้งหมด {saved} รูป ที่: {output_dir}\n")


def create_dataset_structure(dataset_root: str = "dataset"):
    """
    สร้างโครงสร้างโฟลเดอร์สำหรับ YOLO Dataset และสร้าง data.yaml อัตโนมัติ
    """
    dirs = [
        f"{dataset_root}/images/train",
        f"{dataset_root}/images/val",
        f"{dataset_root}/labels/train",
        f"{dataset_root}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    yaml_content = f"""# SafeVision AI Accident Dataset Configuration
path: {os.path.abspath(dataset_root)}
train: images/train
val: images/val

names:
  0: accident
"""
    yaml_path = os.path.join(dataset_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"📁 สร้างโครงสร้าง Dataset และ data.yaml เรียบร้อยที่: {yaml_path}")
    return yaml_path


def train_accident_model(
    data_yaml_path: str = "dataset/data.yaml",
    base_model: str = "yolo11s.pt",
    epochs: int = 60,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",  # ใส่ 'cpu' หากไม่มีการ์ดจอ NVIDIA
):
    """
    สั่ง Train หรือ Fine-tune โมเดลตรวจจับอุบัติเหตุด้วย Ultralytics YOLO
    
    :param data_yaml_path: ไฟล์ config ของ dataset
    :param base_model: โมเดลเริ่มต้น เช่น 'yolo11s.pt', 'yolo11m.pt' หรือ 'car-accident-detection-web/src/model/best.pt'
    :param epochs: จำนวนรอบในการเทรน
    :param imgsz: ขนาดภาพที่ใช้เทรน (ค่ามาตรฐาน 640)
    :param batch_size: ขนาด batch
    :param device: '0' สำหรับ GPU หรือ 'cpu'
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ กรุณาติดตั้ง ultralytics ก่อน: pip install ultralytics")
        return

    print(f"\n🚀 เริ่มต้นการเทรนโมเดลด้วยฐาน: {base_model}")
    print(f"📊 พารามิเตอร์: epochs={epochs}, imgsz={imgsz}, batch={batch_size}, device={device}")

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        # Data Augmentation ที่เหมาะกับคลิปจราจร
        mosaic=1.0,        # ผสม 4 ภาพช่วยให้จับวัตถุขนาดเล็กและมุมกล้องหลากมิติ
        mixup=0.15,        # ผสมภาพซ้อนทับกัน ช่วยให้อ่านการบดบัง (Occlusion) ได้ดี
        perspective=0.0005,# ปรับมุมมองกล้อง CCTV
        flipud=0.0,        # ไม่พลิกภาพกลับหัว (รถไม่วิ่งกลับหัว)
        fliplr=0.5,        # พลิกซ้าย-ขวา 50%
        project="runs/accident_train",
        name="yolo11_accident_exp",
    )

    print("🎉 การเทรนเสร็จสมบูรณ์! น้ำหนักโมเดลที่ดีที่สุดจะอยู่ที่: runs/accident_train/yolo11_accident_exp/weights/best.pt")
    print("👉 คุณสามารถนำไฟล์ best.pt ใหม่นี้ไปวางทับที่ car-accident-detection-web/src/model/best.pt ได้ทันที!")


if __name__ == "__main__":
    print("==================================================")
    print("  SafeVision AI - Model Training Assistant Tool   ")
    print("==================================================")
    print("วิธีใช้งาน:")
    print("1. หากต้องการดึงภาพจากวิดีโอใหม่มาตีกรอบ:")
    print("   extract_frames_from_video('คลิปวิดีโอ.mp4', 'output_frames', sample_interval_sec=1.0)")
    print("")
    print("2. หากต้องการสร้างโฟลเดอร์สำหรับ Labeling:")
    print("   create_dataset_structure('dataset')")
    print("")
    print("3. เมื่อใส่ภาพและไฟล์ .txt ใน dataset แล้ว สั่งเทรนได้ทันที:")
    print("   train_accident_model('dataset/data.yaml', base_model='yolo11s.pt', epochs=60)")
    print("==================================================")
