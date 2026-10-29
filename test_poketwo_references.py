import io
import json
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageOps

# ============================================================
# FILES AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

NAMES_FILE = BASE_DIR / "pokemon_names.json"
ORIGINAL_FEATURES_FILE = BASE_DIR / "pokemon_image_features.pt"
POKETWO_FEATURES_FILE = BASE_DIR / "poketwo_image_features.pt"

# Use the same model as bot.py.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep these weights consistent with the existing detector.
IMAGE_WEIGHT = 0.65
TEXT_WEIGHT = 0.35


# ============================================================
# LOAD POKÉMON NAMES
# ============================================================

def load_pokemon_names():
    with open(NAMES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        try:
            data = dict(
                sorted(data.items(), key=lambda item: int(item[0]))
            )
        except (ValueError, TypeError):
            pass

        names = list(data.values())
    else:
        names = data

    return [
        str(name).replace("-", " ").strip()
        for name in names
    ]


# ============================================================
# IMAGE VARIANTS: SAME AS bot.py
# ============================================================

def make_image_variants(image):
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    variants = [image]

    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2

    variants.append(
        image.crop((left, top, left + side, top + side))
    )

    crop_width = int(width * 0.85)
    crop_height = int(height * 0.85)

    crop_left = (width - crop_width) // 2
    crop_top = (height - crop_height) // 2

    variants.append(
        image.crop((
            crop_left,
            crop_top,
            crop_left + crop_width,
            crop_top + crop_height
        ))
    )

    return variants


# ============================================================
# LOAD FEATURES
# ============================================================

def load_feature_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    features = data["features"].float()
    ids = data["image_ids"].long()

    if len(features) != len(ids):
        raise ValueError(f"Feature/ID count mismatch in {path.name}")

    features = features / features.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    return features, ids.tolist()


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize(values):
    minimum = values.min()
    maximum = values.max()

    if (maximum - minimum).item() < 1e-8:
        return torch.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


# ============================================================
# MAIN TEST
# ============================================================

def main():
    pokemon_names = load_pokemon_names()

    print(f"Pokémon names loaded: {len(pokemon_names)}")
    print(f"Using device: {DEVICE}")

    print("\nLoading original reference features...")
    original_features, original_ids = load_feature_file(
        ORIGINAL_FEATURES_FILE
    )

    print(f"Original reference images: {len(original_ids)}")

    print("\nLoading Pokétwo reference features...")
    poketwo_features, poketwo_ids = load_feature_file(
        POKETWO_FEATURES_FILE
    )

    print(f"Pokétwo reference images: {len(poketwo_ids)}")

    # Combine only in memory; neither file is overwritten.
    combined_features = torch.cat(
        [original_features, poketwo_features],
        dim=0
    )
    combined_ids = original_ids + poketwo_ids

    print(f"Combined reference images: {len(combined_ids)}")

    print("\nLoading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(DEVICE)
    model.eval()

    print("Preparing text features...")
    prompts = [
        (
            f"a clear image of {name}, "
            f"the Pokémon species {name}, "
            f"official Pokémon character artwork"
        )
        for name in pokemon_names
    ]

    text_batches = []

    with torch.inference_mode():
        for start in range(0, len(prompts), 64):
            tokens = tokenizer(
                prompts[start:start + 64]
            ).to(DEVICE)

            text = model.encode_text(tokens)
            text = text / text.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            text_batches.append(text.cpu())

    text_features = torch.cat(text_batches, dim=0)

    # Use the last spawn image saved by the existing detector.
    test_image_path = BASE_DIR / "last_spawn_image.png"

    if not test_image_path.exists():
        raise FileNotFoundError(
            f"Test image not found: {test_image_path}"
        )

    with Image.open(test_image_path) as image:
        variants = make_image_variants(image)

    print(f"\nTest image: {test_image_path.name}")
    print(f"Image variants: {len(variants)}")

    image_scores_per_view = []
    text_scores_per_view = []

    with torch.inference_mode():
        for variant in variants:
            tensor = preprocess(variant).unsqueeze(0).to(DEVICE)

            query = model.encode_image(tensor)
            query = query / query.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            query = query.cpu()

            # Score all original and Pokétwo references.
            image_scores = (
                query @ combined_features.T
            ).squeeze(0)

            # Score the text prompts.
            text_scores = (
                query @ text_features.T
            ).squeeze(0)

            image_scores_per_view.append(image_scores)
            text_scores_per_view.append(text_scores)

    # Best image similarity per Pokémon across all views.
    image_scores_by_id = {}

    for scores in image_scores_per_view:
        for position, pokemon_id in enumerate(combined_ids):
            score = scores[position].item()

            if (
                pokemon_id not in image_scores_by_id
                or score > image_scores_by_id[pokemon_id]
            ):
                image_scores_by_id[pokemon_id] = score

    text_scores = torch.stack(
        text_scores_per_view
    ).max(dim=0).values

    valid_ids = [
        pokemon_id
        for pokemon_id in image_scores_by_id
        if 1 <= pokemon_id <= len(pokemon_names)
    ]

    image_values = torch.tensor([
        image_scores_by_id[pokemon_id]
        for pokemon_id in valid_ids
    ])

    text_values = torch.tensor([
        text_scores[pokemon_id - 1].item()
        for pokemon_id in valid_ids
    ])

    normalized_image = normalize(image_values)
    normalized_text = normalize(text_values)

    combined_scores = (
        IMAGE_WEIGHT * normalized_image
        + TEXT_WEIGHT * normalized_text
    )

    results = []

    for index, pokemon_id in enumerate(valid_ids):
        results.append((
            pokemon_names[pokemon_id - 1].lower(),
            combined_scores[index].item(),
            image_values[index].item(),
            text_values[index].item()
        ))

    results.sort(key=lambda result: result[1], reverse=True)

    print("\n" + "=" * 58)
    print("TOP 5 PREDICTIONS — COMBINED REFERENCES")
    print("=" * 58)

    for rank, (name, combined, image, text) in enumerate(
        results[:5], start=1
    ):
        print(
            f"{rank}. {name} | combined: {combined:.4f} "
            f"| image: {image:.4f} | text: {text:.4f}"
        )

    print("\n" + "=" * 58)
    print("POKÉTWO REFERENCE MATCHES")
    print("=" * 58)

    # Show which Pokémon received the highest similarity to any
    # downloaded Pokétwo image, separately from the combined score.
    poketwo_only_scores = {}

    for scores in image_scores_per_view:
        for position, pokemon_id in enumerate(poketwo_ids):
            score = scores[
                len(original_ids) + position
            ].item()

            if (
                pokemon_id not in poketwo_only_scores
                or score > poketwo_only_scores[pokemon_id]
            ):
                poketwo_only_scores[pokemon_id] = score

    # The same Pokémon has many references, so retain the maximum.
    poketwo_results = []

    for pokemon_id, score in poketwo_only_scores.items():
        if 1 <= pokemon_id <= len(pokemon_names):
            poketwo_results.append((
                pokemon_names[pokemon_id - 1].lower(),
                score
            ))

    poketwo_results.sort(key=lambda item: item[1], reverse=True)

    for rank, (name, score) in enumerate(poketwo_results[:5], start=1):
        print(f"{rank}. {name} | best Pokétwo image similarity: {score:.4f}")

    print("\nTest finished. No existing files were modified.")


if __name__ == "__main__":
    main()
