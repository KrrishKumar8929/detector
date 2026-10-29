import time
import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device).eval()

image = Image.open("desktop_spawn_crop.png").convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.inference_mode():
    start = time.perf_counter()
    result = model.encode_image(image_tensor)
    elapsed = time.perf_counter() - start

print("Device:", device)
print("Image encoding time:", round(elapsed, 4), "seconds")
print("Benchmark complete.")