import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from PIL import Image

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ================= DATASET =================
train_dataset = datasets.ImageFolder("router_dataset/train")
val_dataset = datasets.ImageFolder("router_dataset/validation")

labels = train_dataset.classes
num_labels = len(labels)

print("Labels:", labels)

# ================= MODEL =================
model_name = "facebook/dinov2-small"

processor = AutoImageProcessor.from_pretrained(model_name)

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels
).to(device)

# ================= LoRA =================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)

# ================= COLLATE FUNCTION =================
def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        images.append(inputs["pixel_values"][0])
        labels.append(label)

    return {
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels)
    }

# ================= TRAINING =================
training_args = TrainingArguments(
    output_dir="router_output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# ================= TRAINER =================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

# ================= TRAIN =================
trainer.train()

# ================= SAVE =================
trainer.save_model("router_model_final")

print("✅ Training complete!")