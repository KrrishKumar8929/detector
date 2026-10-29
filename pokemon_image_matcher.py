import os
import json
import torch
import open_clip
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "pokemon_images")
NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")
IMAGE_FILE = os.path.join(BASE_DIR, "selected_pokemon.png")
CACHE_FILE = os.path.join(BASE_DIR, "pokemon_image_features.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Pokémon names...")

with open(NAMES_FILE, "r", encoding="utf-8") as f:
    names_data = json.load(f)

if isinstance(names_data, dict):
    pokemon_names = list(names_data.values())
else:
    pokemon_names = names_data

pokemon_names = [
    str(name).replace("-", " ").strip()
    for name in pokemon_names
]

print("Loading AI model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

if os.path.exists(CACHE_FILE):
    print("Loading saved image features...")

    cache = torch.load(
        CACHE_FILE,
        map_location=device,
        weights_only=True
    )

    if isinstance(cache, dict):
        features = cache["features"]
        image_ids = cache["image_ids"]
    else:
        raise RuntimeError(
            "Your old image cache needs to be rebuilt. "
            "Delete pokemon_image_features.pt and run "
            "pokemon_image_matcher.py again."
        )

else:
    print("Preparing reference images.")
    print("This may take several minutes the first time.")

    all_features = []
    image_ids = []

    with torch.no_grad():
        for pokemon_id in range(1, 1026):
            path = os.path.join(IMAGES_DIR, f"{pokemon_id}.png")

            if not os.path.exists(path):
                continue

            try:
                image = Image.open(path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                feature = model.encode_image(image_tensor)
                feature /= feature.norm(dim=-1, keepdim=True)

                all_features.append(feature.cpu())
                image_ids.append(pokemon_id)

            except Exception as error:
                print(f"Skipping image {pokemon_id}: {error}")

    if not all_features:
        raise RuntimeError("No reference images could be processed.")

features = torch.cat(all_features, dim=0)

torch.save(
    {
        "features": features,
        "image_ids": torch.tensor(image_ids)
    },
    CACHE_FILE
)

print("Reference images prepared and saved.")

if not os.path.exists(IMAGE_FILE):
    raise FileNotFoundError(
        "selected_pokemon.png not found. Select an image first."
    )

print("Analyzing your selected Discord image...")

image = Image.open(IMAGE_FILE).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    query_features = model.encode_image(image_tensor)
    query_features /= query_features.norm(dim=-1, keepdim=True)

    similarities = (query_features.cpu() @ features.T).squeeze(0)

top_count = min(5, len(image_ids))
scores, positions = similarities.topk(top_count)

print("\n========== IMAGE MATCH RESULTS ==========")

for rank, (score, position) in enumerate(
    zip(scores.tolist(), positions.tolist()), start=1
):
    pokemon_id = image_ids[position]

    if pokemon_id <= len(pokemon_names):
        name = pokemon_names[pokemon_id - 1]
    else:
        name = f"Pokémon ID {pokemon_id}"

    print(
        f"{rank}. {name} "
        f"(similarity score: {score:.3f})"
    )

print("\nThese are similarity-based predictions, not guaranteed answers.")