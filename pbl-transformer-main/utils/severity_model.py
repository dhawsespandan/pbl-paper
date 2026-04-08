import torch.serialization
import numpy as np

torch.serialization.add_safe_globals([np.core.multiarray.scalar])
import torch
import torchvision.models as models
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.efficientnet_b4(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    1
)

checkpoint = torch.load(
    "weights/efficientnetb4_spandan.pth",
    map_location=device,
    weights_only=False
)

state_dict = checkpoint["model_state"]

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("backbone."):
        new_key = k.replace("backbone.", "features.")
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

def predict_severity(tensor):
    with torch.no_grad():
        return model(tensor).item() * 100