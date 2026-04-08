import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

model = AutoModelForImageClassification.from_pretrained(
    "router_model_final"
).to(device)

labels = ["flower", "fruit", "leaf"]

def predict_route(image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits).item()
    return labels[pred]