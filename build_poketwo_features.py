import json
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageFile

# Allow partially downloaded images to be opened where possible.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Project paths
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "poketwo_reference_images"
NAMES_FILE = BASE_DIR / "pokemon_names.json"
OUTPUT_FILE = BASE_DIR / "poketwo_image_features.pt"

# Use the same CLIP model as the existing detector.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

def load_pokemon_names():
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        try:
            data = dict(sorted(data.items(), key=lambda item: int(item[0])))
        except (ValueError, TypeError):
            pass
        names = list(data.values())
    else:
        names = data

    return [
        str(name).replace("-", " ").strip().lower()
        for name in names
    ]

def main():
    if not REFERENCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Pokétwo reference folder not found: {REFERENCE_DIR}"
        )

    pokemon_names = load_pokemon_names()

    # Map species names to the IDs used by the existing detector.
    name_to_id = {
        name: index
        for index, name in enumerate(pokemon_names, start=1)
    }

    print(f"Using device: {DEVICE}")
    print("Loading CLIP ViT-B-32...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model = model.to(DEVICE)
    model.eval()

    # The dataset folders should be named after Pokémon species.
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    records = []

    for folder in sorted(REFERENCE_DIR.iterdir()):
        if not folder.is_dir():
            continue

        species = folder.name.replace("-", " ").strip().lower()

        if species not in name_to_id:
            print(f"Skipping unrecognized species folder: {folder.name}")
            continue

        pokemon_id = name_to_id[species]

        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in supported_extensions:
                records.append((path, pokemon_id, species))

    if not records:
        raise RuntimeError(
            "No matching images found. Check the species folder names."
        )

    print(f"Found {len(records)} Pokétwo reference images.")
    print("Extracting image features...")

    all_features = []
    image_ids = []
    failed = 0

    with torch.inference_mode():
        for start in range(0, len(records), BATCH_SIZE):
            batch_records = records[start:start + BATCH_SIZE]
            batch_images = []
            batch_ids = []

            for path, pokemon_id, species in batch_records:
                try:
                    with Image.open(path) as img:
                        image = img.convert("RGB")
                        batch_images.append(preprocess(image))
                        batch_ids.append(pokemon_id)
                except Exception as exc:
                    failed += 1
                    print(f"Could not read {path.name}: {exc}")

            if not batch_images:
                continue

            image_tensor = torch.stack(batch_images).to(DEVICE)
            features = model.encode_image(image_tensor)

            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            all_features.append(features.cpu())
            image_ids.extend(batch_ids)

            processed = min(start + len(batch_records), len(records))
            print(
                f"Processed {processed}/{len(records)} image files...",
                end="\r"
            )

    if not all_features:
        raise RuntimeError("No image features could be extracted.")

    features = torch.cat(all_features, dim=0)

    torch.save(
        {
            "features": features,
            "image_ids": torch.tensor(image_ids, dtype=torch.long),
        },
        OUTPUT_FILE
    )

    print()
    print("\nFeature extraction completed!")
    print(f"Images processed successfully: {len(image_ids)}")
    print(f"Images that failed to load: {failed}")
    print(f"Feature shape: {tuple(features.shape)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
