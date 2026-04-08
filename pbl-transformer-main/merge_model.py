from transformers import AutoModelForImageClassification
from peft import PeftModel

# Load base model
base_model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-small")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "router_output/checkpoint-820")

# 🔥 Merge LoRA into base model
model = model.merge_and_unload()

# Save final model
model.save_pretrained("router_model_final")

print("✅ Full model saved successfully!")