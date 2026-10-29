import torch
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path(__file__).resolve().parent

REFERENCE_FILE = BASE / "poketwo_full_reference_features.pt"
TEST_FILE = BASE / "poketwo_full_test_features.pt"

BATCH_SIZE = 64
NUM_POKEMON = 1025

print("Loading reference features...")
reference = torch.load(REFERENCE_FILE, map_location="cpu", weights_only=True)

print("Loading test features...")
test = torch.load(TEST_FILE, map_location="cpu", weights_only=True)

ref_features = reference["features"].float()
ref_ids = reference["image_ids"].long()

test_features = test["features"].float()
test_ids = test["image_ids"].long()

print(f"Reference images: {len(ref_ids):,}")
print(f"Test images: {len(test_ids):,}")

correct_top1 = 0
correct_top5 = 0

species_results = defaultdict(lambda: [0, 0, 0])
mistakes = Counter()

print("\nTesting accuracy...")

with torch.inference_mode():

    for start in range(0, len(test_ids), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(test_ids))

        batch = test_features[start:end]
        actual_ids = test_ids[start:end]

        # Cosine similarity against all reference images.
        similarities = batch @ ref_features.T

        # Each Pokémon receives its highest reference-image similarity.
        scores = torch.full(
            (len(batch), NUM_POKEMON + 1),
            -float("inf")
        )

        scores.scatter_reduce_(
            1,
            ref_ids.unsqueeze(0).expand(len(batch), -1),
            similarities,
            reduce="amax",
            include_self=True
        )

        predictions = scores[:, 1:].topk(5, dim=1).indices + 1

        for i in range(len(batch)):
            actual = actual_ids[i].item()
            predicted = predictions[i].tolist()

            species_results[actual][0] += 1

            if predicted[0] == actual:
                correct_top1 += 1
                species_results[actual][1] += 1
            else:
                mistakes[(actual, predicted[0])] += 1

            if actual in predicted:
                correct_top5 += 1
                species_results[actual][2] += 1

        print(
            f"Tested {end:,}/{len(test_ids):,}",
            end="\r",
            flush=True
        )

total = len(test_ids)

print("\n")
print("=" * 55)
print("FULL DATASET ACCURACY RESULTS")
print("=" * 55)

print(f"Top-1 accuracy: {correct_top1 / total * 100:.2f}%")
print(f"Top-5 accuracy: {correct_top5 / total * 100:.2f}%")

print("\nSpecific Pokémon results:")

for pokemon_id, name in [
    (137, "Porygon"),
    (917, "Tarountula"),
    (158, "Totodile")
]:
    count, top1, top5 = species_results[pokemon_id]

    if count:
        print(
            f"{name}: "
            f"{top1}/{count} Top-1 ({top1/count*100:.1f}%), "
            f"{top5}/{count} Top-5 ({top5/count*100:.1f}%)"
        )

print("\nMost common incorrect ID predictions:")

for (actual, predicted), count in mistakes.most_common(15):
    print(
        f"Actual ID {actual} -> Predicted ID {predicted}: "
        f"{count} times"
    )

print("\nTesting completed.")