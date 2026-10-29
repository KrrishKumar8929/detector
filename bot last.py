import os
import io
import json
import asyncio

import pyautogui
import pyperclip
import pygetwindow as gw
import time

import aiohttp
import discord
import torch
import open_clip

from PIL import Image, ImageOps
from dotenv import load_dotenv
from plyer import notification


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")
CACHE_FILE = os.path.join(BASE_DIR, "pokemon_image_features.pt")
DEBUG_IMAGE = os.path.join(BASE_DIR, "last_spawn_image.png")

load_dotenv(os.path.join(BASE_DIR, ".env"))
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "DISCORD_BOT_TOKEN is missing from your .env file."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPAWN_PHRASES = [
    "a wild pokémon has appeared",
    "a wild pokemon has appeared",
    "a new wild pokémon has appeared",
    "a new wild pokemon has appeared",
]

# Lower temperature makes score differences more pronounced.
# These are ranking adjustments, not calibrated probabilities.
IMAGE_WEIGHT = 0.65
TEXT_WEIGHT = 0.35

processed_messages = set()

model = None
preprocess = None
tokenizer = None

pokemon_names = []
image_features = None
image_ids = []
text_features = None


# ==========================================
# LOAD MODEL AND REFERENCE DATA
# ==========================================

def load_recognition_model():
    global model, preprocess, tokenizer
    global pokemon_names, image_features, image_ids, text_features

    print("Loading Pokémon names...")

    with open(NAMES_FILE, "r", encoding="utf-8") as file:
        names_data = json.load(file)

    if isinstance(names_data, dict):
        # Ensure dictionary keys are ordered by numeric Pokémon ID.
        try:
            names_data = dict(
                sorted(names_data.items(), key=lambda item: int(item[0]))
            )
        except (ValueError, TypeError):
            pass

        pokemon_names = list(names_data.values())
    else:
        pokemon_names = names_data

    pokemon_names = [
        str(name).replace("-", " ").strip()
        for name in pokemon_names
    ]

    print(f"Loaded {len(pokemon_names)} Pokémon names.")
    print(f"Using device: {DEVICE}")

    print("Loading CLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(DEVICE)
    model.eval()

    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"Reference cache not found:\n{CACHE_FILE}"
        )

    print("Loading reference-image features...")

    cache = torch.load(
        CACHE_FILE,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(cache, dict):
        raise RuntimeError(
            "Unsupported feature cache. Rebuild the cache first."
        )

    image_features = cache["features"].float()
    image_ids = cache["image_ids"].long().tolist()

    if len(image_features) != len(image_ids):
        raise RuntimeError(
            "Reference features and Pokémon IDs do not match."
        )

    image_features = image_features / (
        image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    )

    print(f"Loaded {len(image_ids)} reference images.")

    print("Preparing text features...")

    prompts = [
        (
            f"a clear image of {name}, "
            f"the Pokémon species {name}, "
            f"official Pokémon character artwork"
        )
        for name in pokemon_names
    ]

    batches = []

    with torch.inference_mode():
        for start in range(0, len(prompts), 64):
            tokens = tokenizer(
                prompts[start:start + 64]
            ).to(DEVICE)

            features = model.encode_text(tokens)

            features = features / (
                features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            )

            batches.append(features.cpu())

    text_features = torch.cat(batches, dim=0)

    print("Recognition system loaded successfully!")


# ==========================================
# IMAGE PREPARATION
# ==========================================

def make_image_variants(image):
    """
    Create a small set of views of the same spawn image.

    The original image is retained so that a crop cannot remove
    important parts of the Pokémon.
    """

    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size

    variants = [image]

    # Add a centered square crop.
    side = min(width, height)

    left = (width - side) // 2
    top = (height - side) // 2

    square = image.crop((
        left,
        top,
        left + side,
        top + side
    ))

    variants.append(square)

    # Add a slightly wider crop to preserve extended body parts.
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


# ==========================================
# RECOGNIZE POKÉMON
# ==========================================

def recognize_image(image_bytes):
    original = Image.open(io.BytesIO(image_bytes))
    variants = make_image_variants(original)

    image_scores_per_view = []
    text_scores_per_view = []

    with torch.inference_mode():
        for variant in variants:
            tensor = preprocess(variant).unsqueeze(0).to(DEVICE)

            query = model.encode_image(tensor)

            query = query / (
                query.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            )

            query = query.cpu()

            image_scores = (query @ image_features.T).squeeze(0)
            text_scores = (query @ text_features.T).squeeze(0)

            # Several reference images may map to the same Pokémon ID.
            # Keep the highest image score for each Pokémon.
            scores_by_id = {}

            for position, pokemon_id in enumerate(image_ids):
                score = image_scores[position].item()

                if (
                    pokemon_id not in scores_by_id
                    or score > scores_by_id[pokemon_id]
                ):
                    scores_by_id[pokemon_id] = score

            image_scores_per_view.append(scores_by_id)
            text_scores_per_view.append(text_scores)

    # Combine the image scores from all views.
    image_scores_by_id = {}

    for scores in image_scores_per_view:
        for pokemon_id, score in scores.items():
            if (
                pokemon_id not in image_scores_by_id
                or score > image_scores_by_id[pokemon_id]
            ):
                image_scores_by_id[pokemon_id] = score

    # Use the best text score across image variants.
    # Text embeddings are identical between variants, but retaining
    # the same structure makes the scoring process explicit.
    text_scores = torch.stack(text_scores_per_view).max(dim=0).values

    # Normalize each score family relative to the candidates.
    # This improves ranking consistency but is not a probability.
    valid_ids = [
        pokemon_id
        for pokemon_id in image_scores_by_id
        if 1 <= pokemon_id <= len(pokemon_names)
    ]

    if not valid_ids:
        return []

    image_values = torch.tensor(
        [image_scores_by_id[pokemon_id] for pokemon_id in valid_ids],
        dtype=torch.float32
    )

    text_values = torch.tensor(
        [text_scores[pokemon_id - 1].item() for pokemon_id in valid_ids],
        dtype=torch.float32
    )

    def normalize(values):
        minimum = values.min()
        maximum = values.max()

        if (maximum - minimum).item() < 1e-8:
            return torch.zeros_like(values)

        return (values - minimum) / (maximum - minimum)

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

    return results[:5]

# ==========================================
# COPY COMMAND AND FOCUS DISCORD
# ==========================================

def prepare_catch_command(command):
    """Copy the catch command and bring Discord Desktop forward."""

    try:
        # Copy the command to the clipboard.
        pyperclip.copy(command)
        print(f"Copied to clipboard: {command}")

        # Find an open Discord Desktop window.
        windows = gw.getAllWindows()

        discord_windows = [
            window for window in windows
            if window.title
            and "discord" in window.title.lower()
        ]

        if not discord_windows:
            print("Discord window not found. Command is still copied.")
            return

        discord_window = discord_windows[0]

        # Restore Discord if minimized.
        if discord_window.isMinimized:
            discord_window.restore()
            time.sleep(0.5)

        # Bring Discord to the foreground.
        discord_window.activate()
        time.sleep(0.5)

        print("Discord brought to the foreground.")
        print("Paste with Ctrl+V, then press Enter.")

    except Exception as error:
        print(f"Clipboard/Discord focus error: {error}")


# ==========================================
# DISCORD CLIENT
# ==========================================

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print()
    print("=" * 45)
    print(f"Connected as: {client.user}")
    print("Pokémon Background Detector is running.")
    print("Waiting for Pokémon spawns...")
    print("=" * 45)


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.id in processed_messages:
        return

    text_parts = [message.content.lower()]

    for embed in message.embeds:
        text_parts.append((embed.title or "").lower())
        text_parts.append((embed.description or "").lower())

        for field in embed.fields:
            text_parts.append(field.name.lower())
            text_parts.append(field.value.lower())

    full_text = " ".join(text_parts)

    if not any(phrase in full_text for phrase in SPAWN_PHRASES):
        return

    processed_messages.add(message.id)

    print()
    print("=" * 45)
    print("POKÉMON SPAWN DETECTED!")

    guild_name = message.guild.name if message.guild else "Direct Message"
    channel_name = getattr(message.channel, "name", "DM")

    print(f"Server: {guild_name}")
    print(f"Channel: #{channel_name}")

    image_url = None

    for attachment in message.attachments:
        if (
            attachment.content_type
            and attachment.content_type.startswith("image/")
        ):
            image_url = attachment.url
            break

    if not image_url:
        for embed in message.embeds:
            if embed.image and embed.image.url:
                image_url = embed.image.url
                break

            if embed.thumbnail and embed.thumbnail.url:
                image_url = embed.thumbnail.url
                break

    if not image_url:
        print("No image found.")
        return

    print("Downloading Pokémon image...")

    try:
        timeout = aiohttp.ClientTimeout(total=20)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    print(
                        f"Image download failed: HTTP {response.status}"
                    )
                    return

                image_bytes = await response.read()

        with open(DEBUG_IMAGE, "wb") as file:
            file.write(image_bytes)

        print(f"Image saved: {DEBUG_IMAGE}")

    except Exception as error:
        print(f"Download error: {error}")
        return

    print("Recognizing Pokémon...")

    try:
        results = await asyncio.to_thread(
            recognize_image,
            image_bytes
        )
    except Exception as error:
        print(f"Recognition error: {error}")
        return

    if not results:
        print("No predictions returned.")
        return

    print()
    print("TOP 5 PREDICTIONS")

    for rank, (name, score, image_score, text_score) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank}. {name} "
            f"| combined: {score:.4f} "
            f"| image: {image_score:.3f} "
            f"| text: {text_score:.3f}"
        )

    best_name = results[0][0]
    catch_command = f"@Pokétwo#8236 c {best_name}"

    print()
    print(f"Best prediction: {best_name}")
    print(f"Prepared command: {catch_command}")

    await asyncio.to_thread(
        prepare_catch_command,
        catch_command
    )

    print("The command has NOT been sent automatically.")
    print("=" * 45)

    try:
        notification.notify(
            title=f"Pokémon detected: {best_name}",
            message=f"Prepared command: {catch_command}",
            timeout=8
        )
    except Exception as error:
        print(f"Notification error: {error}")


# ==========================================
# START
# ==========================================

if __name__ == "__main__":
    load_recognition_model()
    client.run(TOKEN)