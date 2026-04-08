from ultralytics import YOLO

model = YOLO("weights/yolo26ncls_anindita.pt")

def predict_fruit(image_path):
    res = model(image_path)

    probs = res[0].probs
    pred = probs.top1
    conf = probs.top1conf.item()

    label = res[0].names[pred]

    return label, conf