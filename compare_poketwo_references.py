import json
from pathlib import Path

import torch
import open_clip
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent

NAMES_FILE = BASE_DIR / "pokemon_names.json"
ORIGINAL_FILE = BASE_DIR / "pokemon_image_features.pt"
POKETWO_FILE = BASE_DIR / "poketwo_image_features.pt"
TEST_IMAGE = BASE_DIR / "last_spawn_image.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WEIGHT = 0.65
TEXT_WEIGHT = 0.35


def load_names():
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = dict(
            sorted(data.items(), key=lambda item: int(item[0]))
        )
        data = list(data.values())

    return [str(name).replace("-", " ").strip() for name in data]


def load_features(path):
    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    features = data["features"].float()
    ids = data["image_ids"].long().tolist()

    features = features / features.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    return features, ids


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


def normalize(values):
    minimum = values.min()
    maximum = values.max()

    if (maximum - minimum).item() < 1e-8:
        return torch.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def rank_predictions(
    query_features,
    text_features,
    reference_features,
    reference_ids,
    pokemon_names
):
    # Keep the highest reference-image score for each species.
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

    # Best text score across the image variants.
    text_scores = torch.stack([
        (query @ text_features.T).squeeze(0)
        for query in query_features
    ]).max(dim=0).values

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

    combined = (
        IMAGE_WEIGHT * normalize(image_values)
        + TEXT_WEIGHT * normalize(text_values)
    )

    results = []

    for index, pokemon_id in enumerate(valid_ids):
        results.append((
            pokemon_names[pokemon_id - 1].lower(),
            combined[index].item(),
            image_values[index].item(),
            text_values[index].item()
        ))

    results.sort(key=lambda row: row[1], reverse=True)
    return results[:5]


def print_results(title, results):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)

    for rank, (name, combined, image, text) in enumerate(
        results, start=1
    ):
        print(
            f"{rank}. {name} | combined: {combined:.4f} "
            f"| image: {image:.4f} | text: {text:.4f}"
        )


def main():
    for path in (
        NAMES_FILE,
        ORIGINAL_FILE,
        POKETWO_FILE,
        TEST_IMAGE
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    names = load_names()

    print("Loading original references...")
    original_features, original_ids = load_features(ORIGINAL_FILE)

    print("Loading Pokétwo references...")
    poketwo_features, poketwo_ids = load_features(POKETWO_FILE)

    print(f"Original references: {len(original_ids)}")
    print(f"Pokétwo references: {len(poketwo_ids)}")

    print("\nLoading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(DEVICE)
    model.eval()

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

    query_features = []

    with torch.inference_mode():
        for variant in variants:
            tensor = preprocess(variant).unsqueeze(0).to(DEVICE)

            query = model.encode_image(tensor)
            query = query / query.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            query_features.append(query.cpu())

    # Test A: Original references only.
    original_results = rank_predictions(
        query_features,
        text_features,
        original_features,
        original_ids,
        names
    )

    # Test B: Original plus Pokétwo references.
    combined_features = torch.cat(
        [original_features, poketwo_features],
        dim=0
    )
    combined_ids = original_ids + poketwo_ids

    combined_results = rank_predictions(
        query_features,
        text_features,
        combined_features,
        combined_ids,
        names
    )

    print_results(
        "TEST A — ORIGINAL REFERENCES ONLY",
        original_results
    )

    print_results(
        "TEST B — ORIGINAL + POKÉTWO REFERENCES",
        combined_results
    )

    print("\nActual Pokémon: Slakoth")
    print("Comparison complete. No existing files were modified.")


if __name__ == "__main__":
    main()