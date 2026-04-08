from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("router_output/checkpoint-820")
model.save_pretrained("router_model_final")

print("✅ Model saved successfully!")