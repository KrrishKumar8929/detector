import json
from pathlib import Path
from collections import defaultdict

import torch
import open_clip
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
NAMES_FILE = BASE_DIR / "pokemon_names.json"
FEATURES_FILE = BASE_DIR / "pokemon_image_features.pt"
REFERENCE_DIR = BASE_DIR / "poketwo_reference_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = [
    ("A: Current 65/35", 0.65, 0.35),
    ("B: Image-focused 80/20", 0.80, 0.20),
    ("C: Image-focused 90/10", 0.90, 0.10),
    ("D: Image only 100/0", 1.00, 0.00),
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_names():
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        try:
            data = dict(sorted(data.items(), key=lambda x: int(x[0])))
        except (ValueError, TypeError):
            pass
        data = list(data.values())

    return [str(n).replace("-", " ").strip().lower() for n in data]


def load_features():
    data = torch.load(
        FEATURES_FILE,
        map_location="cpu",
        weights_only=True
    )
    features = data["features"].float()
    ids = data["image_ids"].long()
    features = features / features.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)
    return features, ids.tolist()


def make_variants(image):
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
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    variants.append(
        image.crop((
            left, top,
            left + crop_width, top + crop_height
        ))
    )

    return variants


def normalize(values):
    low = values.min()
    high = values.max()
    if (high - low).item() < 1e-8:
        return torch.zeros_like(values)
    return (values - low) / (high - low)


def main():
    names = load_names()
    features, ids = load_features()

    print(f"Pokémon species: {len(names)}")
    print(f"Original reference images: {len(ids)}")
    print(f"Device: {DEVICE}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(DEVICE).eval()

    print("Preparing text features...")

    prompts = [
        f"a clear image of {name}, the Pokémon species {name}, "
        f"official Pokémon character artwork"
        for name in names
    ]

    text_batches = []
    with torch.inference_mode():
        for start in range(0, len(prompts), 64):
            tokens = tokenizer(prompts[start:start + 64]).to(DEVICE)
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            text_batches.append(encoded.cpu())

    text_features = torch.cat(text_batches, dim=0)

    # Map each Pokémon name to the IDs used by the original database.
    name_to_id = {
        name: index
        for index, name in enumerate(names, start=1)
    }

    # Prepare test images and their known labels.
    tests = []
    for folder_name in ["porygon", "tarountula", "totodile"]:
        folder = REFERENCE_DIR / folder_name
        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        paths = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
            and ".cache" not in p.parts
        )

        expected_id = name_to_id.get(folder_name)
        if expected_id is None:
            print(f"Could not find species ID for {folder_name}")
            continue

        tests.extend((path, folder_name, expected_id) for path in paths)

    if not tests:
        print("No test images found.")
        return

    print(f"\nTesting {len(tests)} images...")
    print("This may take several minutes on CPU.\n")

    # Collect normalized scores for each image.
    results_by_weight = {
        label: [] for label, _, _ in WEIGHTS
    }

    correct_top1_by_species = defaultdict(lambda: defaultdict(int))
    top5_by_species = defaultdict(lambda: defaultdict(int))
    counts_by_species = defaultdict(int)

    with torch.inference_mode():
        for number, (path, species, expected_id) in enumerate(tests, 1):
            try:
                with Image.open(path) as image:
                    variants = make_variants(image)

                image_views = []
                text_views = []

                for variant in variants:
                    tensor = preprocess(variant).unsqueeze(0).to(DEVICE)
                    query = model.encode_image(tensor)
                    query = query / query.norm(
                        dim=-1, keepdim=True
                    ).clamp(min=1e-12)
                    query = query.cpu()

                    image_views.append((query @ features.T).squeeze(0))
                    text_views.append((query @ text_features.T).squeeze(0))

                # Max image similarity per species ID across all views.
                best_image = {}
                for view_scores in image_views:
                    for pos, pokemon_id in enumerate(ids):
                        score = view_scores[pos].item()
                        if pokemon_id not in best_image or score > best_image[pokemon_id]:
                            best_image[pokemon_id] = score

                valid_ids = [
                    pokemon_id for pokemon_id in best_image
                    if 1 <= pokemon_id <= len(names)
                ]

                image_values = torch.tensor([
                    best_image[pokemon_id] for pokemon_id in valid_ids
                ])
                text_scores = torch.stack(text_views).max(dim=0).values
                text_values = torch.tensor([
                    text_scores[pokemon_id - 1].item()
                    for pokemon_id in valid_ids
                ])

                norm_image = normalize(image_values)
                norm_text = normalize(text_values)

                counts_by_species[species] += 1

                for label, image_weight, text_weight in WEIGHTS:
                    combined = (
                        image_weight * norm_image
                        + text_weight * norm_text
                    )
                    order = torch.argsort(combined, descending=True)
                    ranked_ids = [
                        valid_ids[i] for i in order.tolist()
                    ]

                    top1_correct = ranked_ids[0] == expected_id
                    top5_correct = expected_id in ranked_ids[:5]

                    results_by_weight[label].append(
                        (top1_correct, top5_correct)
                    )

                    if top1_correct:
                        correct_top1_by_species[label][species] += 1

                    if top5_correct:
                        top5_by_species[label][species] += 1

                    # Print incorrect Totodile predictions only
                    if (
                        species == "totodile"
                        and label == "A: Current 65/35"
                        and not top1_correct
                        and number <= 210
                    ):
                        predicted = names[ranked_ids[0] - 1]
                        top_five = [
                            names[pokemon_id - 1]
                            for pokemon_id in ranked_ids[:5]
                        ]

                        print(
                            f"\nTotodile image: {path.name}"
                            f"\n  Predicted #1: {predicted}"
                            f"\n  Top 5: {', '.join(top_five)}"
                        )
                    

                    if top1_correct:
                        correct_top1_by_species[label][species] += 1
                    if top5_correct:
                        top5_by_species[label][species] += 1

            except Exception as error:
                print(f"Skipped {path.name}: {error}")

            if number % 15 == 0 or number == len(tests):
                print(f"Processed {number}/{len(tests)} images")

    print("\n" + "=" * 68)
    print("ACCURACY SUMMARY")
    print("=" * 68)

    for label, _, _ in WEIGHTS:
        results = results_by_weight[label]
        if not results:
            continue

        total = len(results)
        top1 = sum(x[0] for x in results)
        top5 = sum(x[1] for x in results)

        print(f"\n{label}")
        print(f"  Top-1: {top1}/{total} ({100 * top1 / total:.1f}%)")
        print(f"  Top-5: {top5}/{total} ({100 * top5 / total:.1f}%)")

        for species in ["porygon", "tarountula", "totodile"]:
            count = counts_by_species[species]
            if count:
                s1 = correct_top1_by_species[label][species]
                s5 = top5_by_species[label][species]
                print(
                    f"  {species.title()}: "
                    f"top-1 {s1}/{count}, top-5 {s5}/{count}"
                )

    print("\nTest finished. No existing files were modified.")


if __name__ == "__main__":
    main()
