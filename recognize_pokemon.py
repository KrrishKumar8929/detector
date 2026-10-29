import json
import os
import torch
import open_clip
from PIL import Image

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Files
NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")
IMAGE_FILE = os.path.join(BASE_DIR, "selected_pokemon.png")

# Use CPU or GPU automatically
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Pokémon names...")

with open(NAMES_FILE, "r", encoding="utf-8") as file:
    pokemon_names = json.load(file)

# Support either a list of names or a dictionary of names
if isinstance(pokemon_names, dict):
    pokemon_names = list(pokemon_names.values())

pokemon_names = [
    str(name).replace("-", " ").strip()
    for name in pokemon_names
]

print(f"Loaded {len(pokemon_names)} Pokémon names.")

print("Loading AI model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

print("Preparing Pokémon names for AI recognition...")

prompts = [
    f"a picture of the Pokémon {name}"
    for name in pokemon_names
]

all_text_features = []

# Process names in small batches to avoid using too much memory
with torch.no_grad():
    for start in range(0, len(prompts), 64):
        batch_prompts = prompts[start:start + 64]
        tokens = tokenizer(batch_prompts).to(device)

        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )

        all_text_features.append(text_features)

text_features = torch.cat(all_text_features, dim=0)

print("\nAI is ready!")

if not os.path.exists(IMAGE_FILE):
    print("\nNo selected Pokémon image was found.")
    print("First, use your screenshot selection program.")
    raise SystemExit

print("\nAnalyzing selected image...")

try:
    image = Image.open(IMAGE_FILE).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(
            dim=-1, keepdim=True
        )

        similarities = image_features @ text_features.T

        # Get the five highest-scoring predictions
        top_count = min(5, len(pokemon_names))
        scores, indices = similarities[0].topk(top_count)

    print("\n========== PREDICTIONS ==========")

    for rank, (score, index) in enumerate(
        zip(scores.tolist(), indices.tolist()), start=1
    ):
        print(
            f"{rank}. {pokemon_names[index]} "
            f"(similarity score: {score:.3f})"
        )

    print("\nThese are AI predictions, not guaranteed answers.")

except Exception as error:
    print("\nCould not analyze the image.")
    print("Error:", error)