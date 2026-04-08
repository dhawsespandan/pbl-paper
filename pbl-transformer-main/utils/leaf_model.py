import torch
import torchvision.models as models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_v2_s(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)

checkpoint = torch.load(
    "weights/efficientnetb0_astha.pt",
    map_location=device
)

# handle formats
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

def predict_leaf(tensor):
    with torch.no_grad():
        out = model(tensor)
    return torch.argmax(out, dim=1).item()