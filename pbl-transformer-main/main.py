from fastapi import FastAPI, UploadFile, File
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils.router import predict_route
from utils.fruit_model import predict_fruit
from utils.flower_model import count_flowers
from utils.leaf_model import predict_leaf
from utils.severity_model import predict_severity

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path).convert("RGB")

    # -------- ROUTER --------
    task = predict_route(image)

    # -------- FRUIT --------
    if task == "fruit":
        label, conf = predict_fruit(file_path)

        result = {
            "type": "fruit",
            "disease": label,
            "confidence": conf
        }

        if label != "healthy":
            img = transform(image).unsqueeze(0).to(device)
            sev = predict_severity(img)
            result["severity"] = round(sev, 2)

    # -------- LEAF --------
    elif task == "leaf":
        img = transform(image).unsqueeze(0).to(device)
        pred = predict_leaf(img)

        result = {
            "type": "leaf",
            "disease": str(pred)
        }

    # -------- FLOWER --------
    elif task == "flower":
        count = count_flowers(file_path)

        result = {
            "type": "flower",
            "count": count
        }

    else:
        result = {"error": "Unknown"}

    return result