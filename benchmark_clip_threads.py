import time
import statistics
import torch
import open_clip
from PIL import Image

print("Loading CLIP model...")

device = "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device).eval()

image = Image.open("desktop_spawn_crop.png").convert("RGB")
image_tensor = preprocess(image).unsqueeze(0)

thread_counts = [1, 2, 4, 6, 8, 12]

for thread_count in thread_counts:
    torch.set_num_threads(thread_count)

    with torch.inference_mode():
        for _ in range(2):
            model.encode_image(image_tensor)

        timings = []

        for _ in range(5):
            start = time.perf_counter()
            model.encode_image(image_tensor)
            timings.append(time.perf_counter() - start)

    average = statistics.median(timings)

    print(
        "Threads:", thread_count,
        "| Median encoding time:",
        round(average, 4), "seconds"
    )

print("Benchmark complete. Detector files were not changed.")
