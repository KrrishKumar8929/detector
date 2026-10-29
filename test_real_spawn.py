import json
from pathlib import Path

import torch
import open_clip
from PIL import Image

BASE = Path(__file__).resolve().parent

IMAGE_FILE = BASE / "last_spawn_image.png"
FEATURE_FILE = BASE / "poketwo_full_reference_features.pt"
NAMES_FILE = BASE / "pokemon_names.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Pokémon names...")

with open(NAMES_FILE, "r", encoding="utf-8") as f:
    names = json.load(f)

print("Loading new reference database...")

database = torch.load(
    FEATURE_FILE,
    map_location="cpu",
    weights_only=True
)

reference_features = database["features"].float()
reference_ids = database["image_ids"].long()

print(f"Reference images: {len(reference_ids):,}")

print("Loading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(DEVICE)
model.eval()

print("Processing actual Discord spawn image...")

with Image.open(IMAGE_FILE) as img:
    image = preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)

with torch.inference_mode():

    spawn_features = model.encode_image(image).float()

    spawn_features = spawn_features / spawn_features.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    similarities = spawn_features.cpu() @ reference_features.T

    scores = torch.full(
        (1, 1026),
        -float("inf")
    )

    scores.scatter_reduce_(
        1,
        reference_ids.unsqueeze(0),
        similarities,
        reduce="amax",
        include_self=True
    )

    top_scores, top_indices = scores[:, 1:].topk(10, dim=1)

print("\n" + "=" * 55)
print("REAL DISCORD SPAWN TEST")
print("=" * 55)

for rank, (index, score) in enumerate(
    zip(top_indices[0], top_scores[0]),
    start=1
):
    pokemon_id = index.item() + 1
    name = names.get(str(pokemon_id), "Unknown")

    print(
        f"{rank:2}. {name.lower():25} "
        f"similarity: {score.item():.4f}"
    )

print("\nTest completed.")