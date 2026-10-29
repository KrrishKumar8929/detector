import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "pokemon_images")
NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")

with open(NAMES_FILE, "r", encoding="utf-8") as file:
    names = json.load(file)

if isinstance(names, dict):
    names = list(names.values())

missing = []

for pokemon_id in range(1, 1026):
    image_path = os.path.join(
        IMAGES_DIR,
        f"{pokemon_id}.png"
    )

    if not os.path.exists(image_path):
        missing.append(pokemon_id)

print("Pokémon names in database:", len(names))
print("Missing reference images:", len(missing))

if missing:
    print("\nMissing image IDs:")
    print(missing)
else:
    print("\nAll numbered reference images are present.")