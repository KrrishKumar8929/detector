import json
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent

NAMES_FILE = BASE_DIR / "pokemon_names.json"
FEATURES_FILE = BASE_DIR / "pokemon_image_features.pt"
TEST_IMAGE = BASE_DIR / "last_spawn_image.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Each pair is (image weight, text weight).
WEIGHT_CONFIGURATIONS = [
    ("A - Current: 65% image / 35% text", 0.65, 0.35),
    ("B - Image-focused: 80% image / 20% text", 0.80, 0.20),
    ("C - Strongly image-focused: 90% image / 10% text", 0.90, 0.10),
    ("D - Image only: 100% image / 0% text", 1.00, 0.00),
]


def load_names():
    with open(NAMES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = dict(
            sorted(data.items(), key=lambda item: int(item[0]))
        )
        data = list(data.values())

    return [
        str(name).replace("-", " ").strip()
        for name in data
    ]


def load_reference_features():
    data = torch.load(
        FEATURES_FILE,
        map_location="cpu",
        weights_only=True
    )

    features = data["features"].float()
    ids = data["image_ids"].long().tolist()

    if len(features) != len(ids):
        raise ValueError("Reference features and IDs do not match.")

    features = features / features.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    return features, ids


def make_image_variants(image):
    # Match the current detector's image preparation.
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


def normalize(values):
    minimum = values.min()
    maximum = values.max()

    if (maximum - minimum).item() < 1e-8:
        return torch.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def main():
    for path in (NAMES_FILE, FEATURES_FILE, TEST_IMAGE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    names = load_names()

    print(f"Pokémon names: {len(names)}")
    print(f"Using device: {DEVICE}")

    reference_features, reference_ids = load_reference_features()
    print(f"Original reference images: {len(reference_ids)}")

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
        for name in names
    ]

    text_batches = []

    with torch.inference_mode():
        for start in range(0, len(prompts), 64):
            tokens = tokenizer(
                prompts[start:start + 64]
            ).to(DEVICE)

            features = model.encode_text(tokens)

            features = features / features.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            text_batches.append(features.cpu())

    text_features = torch.cat(text_batches, dim=0)

    with Image.open(TEST_IMAGE) as image:
        variants = make_image_variants(image)

    print(f"\nTest image: {TEST_IMAGE.name}")
    print(f"Image variants: {len(variants)}")
    print("\nEncoding spawn image...")

    query_features = []

    with torch.inference_mode():
        for variant in variants:
            tensor = preprocess(variant).unsqueeze(0).to(DEVICE)

            query = model.encode_image(tensor)

            query = query / query.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            query_features.append(query.cpu())

    # Calculate the highest image similarity per Pokémon ID.
    image_scores_by_id = {}

    for query in query_features:
        scores = (query @ reference_features.T).squeeze(0)

        for position, pokemon_id in enumerate(reference_ids):
            score = scores[position].item()

            if (
                pokemon_id not in image_scores_by_id
                or score > image_scores_by_id[pokemon_id]
            ):
                image_scores_by_id[pokemon_id] = score

    text_scores = torch.stack([
        (query @ text_features.T).squeeze(0)
        for query in query_features
    ]).max(dim=0).values

    valid_ids = [
        pokemon_id
        for pokemon_id in image_scores_by_id
        if 1 <= pokemon_id <= len(names)
    ]

    image_values = torch.tensor([
        image_scores_by_id[pokemon_id]
        for pokemon_id in valid_ids
    ])

    text_values = torch.tensor([
        text_scores[pokemon_id - 1].item()
        for pokemon_id in valid_ids
    ])

    # Normalize once so every configuration uses identical inputs.
    normalized_image = normalize(image_values)
    normalized_text = normalize(text_values)

    results_by_weight = {}

    for label, image_weight, text_weight in WEIGHT_CONFIGURATIONS:
        combined = (
            image_weight * normalized_image
            + text_weight * normalized_text
        )

        results = []

        for index, pokemon_id in enumerate(valid_ids):
            results.append((
                names[pokemon_id - 1].lower(),
                combined[index].item(),
                image_values[index].item(),
                text_values[index].item()
            ))

        results.sort(key=lambda row: row[1], reverse=True)
        results_by_weight[label] = results[:10]

    for label, _, _ in WEIGHT_CONFIGURATIONS:
        print("\n" + "=" * 65)
        print(label)
        print("=" * 65)

        for rank, (name, combined, image, text) in enumerate(
            results_by_weight[label][:5], start=1
        ):
            print(
                f"{rank}. {name} | combined: {combined:.4f} "
                f"| image: {image:.4f} | text: {text:.4f}"
            )

    print("\nActual Pokémon: Slakoth")
    print("\nTest finished. No existing files were modified.")


if __name__ == "__main__":
    main()
