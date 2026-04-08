from ultralytics import YOLO

model = YOLO("weights/yolo26m_abhirami.pt")

def count_flowers(image_path):
    res = model(image_path)
    return len(res[0].boxes)