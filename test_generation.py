import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

alias = "aehrc/cxrmate-2"

url = (
    "https://prod-images-static.radiopaedia.org/images/"
    "220869/76052f7902246ff862f52f5d3cd9cd_big_gallery.jpg"
)

print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    alias,
    trust_remote_code=True
)

print("Loading FP16 model...")
model = AutoModelForCausalLM.from_pretrained(
    alias,
    trust_remote_code=True,
    dtype=torch.float16
)

model.eval()

print("Model loaded.")
print("Preparing image...")

processed = processor(images=url)
processed = processed.to(device="cpu")

# CXRMate-2's custom generation code creates time_deltas as FP32,
# but the model is loaded in FP16. Match the model dtype.
if "time_deltas" in processed:
    processed["time_deltas"] = processed["time_deltas"].to(torch.float16)

print("Input prepared.")
print("pixel_values:", processed["pixel_values"].shape)
print("Starting generation...")
print("This may take a while on CPU.")

generation_config = GenerationConfig.from_pretrained(
    alias,
    trust_remote_code=True
)

with torch.no_grad():
    generated_ids = model.generate(
        **processed,
        generation_config=generation_config
    )

print("Generation completed.")

findings, impression = processor.split_and_decode_sections(
    generated_ids
)

print("\n========== FINDINGS ==========")
print(findings)

print("\n========== IMPRESSION ==========")
print(impression)