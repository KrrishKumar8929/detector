import os
import io
import json
import asyncio
import random

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
MULTILINGUAL_FILE = os.path.join(BASE_DIR, "pokemon_multilingual_names.json")
if not os.path.exists(MULTILINGUAL_FILE):
    MULTILINGUAL_FILE = os.path.join(BASE_DIR, "pokemon_multilingual_names(1).json")
ALT_NAMES_FILE = os.path.join(BASE_DIR, "pokemon_user_alt_names.json")
if not os.path.exists(ALT_NAMES_FILE):
    ALT_NAMES_FILE = os.path.join(BASE_DIR, "pokemon_user_alt_names(1).json")
SETTINGS_FILE = os.path.join(BASE_DIR, "pokemon_bot_settings.json")
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

multilingual_by_id = {}
alt_names_by_slug = {}
NAME_MODES = [
    "Current Naming Rules",
    "English",
    "Alternate English",
    "German",
    "French",
    "Japanese Romanized",
    "Japanese Script",
    "All Available Languages",
    "Random Language",
    "My Alt Names (TXT + Current Rules)",
]
NAME_MODE = "Current Naming Rules"
MESSAGE_DELAY_SECONDS = 2.0  # Visible, fixed delay before posting the name.


# ==========================================
# NAME OPTIONS AND SETTINGS
# ==========================================

def load_name_data():
    """Load optional multilingual and custom-name files without changing them."""
    global multilingual_by_id, alt_names_by_slug

    if os.path.exists(MULTILINGUAL_FILE):
        try:
            with open(MULTILINGUAL_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            for record in data.get("records", []):
                try:
                    pokemon_id = int(record.get("id", record.get("dex_number", 0)))
                except (TypeError, ValueError):
                    continue
                if pokemon_id > 0:
                    multilingual_by_id[pokemon_id] = record
            print(f"Loaded multilingual names for {len(multilingual_by_id)} entries.")
        except Exception as error:
            print(f"Could not load multilingual names: {error}")

    if os.path.exists(ALT_NAMES_FILE):
        try:
            with open(ALT_NAMES_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
            alt_names_by_slug = data.get("by_slug", {}) if isinstance(data, dict) else {}
            print(f"Loaded custom-name rules for {len(alt_names_by_slug)} entries.")
        except Exception as error:
            print(f"Could not load custom alternate names: {error}")


def choose_name_mode():
    """Choose the output-name style in the console and remember it in settings."""
    global NAME_MODE, MESSAGE_DELAY_SECONDS
    settings = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
                settings = json.load(file)
        except Exception:
            settings = {}

    saved_mode = settings.get("name_mode", "Current Naming Rules")
    print("\nPokémon name output mode:")
    for index, mode in enumerate(NAME_MODES, start=1):
        marker = " (saved)" if mode == saved_mode else ""
        print(f"  {index}. {mode}{marker}")
    choice = input(f"Choose 1-{len(NAME_MODES)} or press Enter to keep [{saved_mode}]: ").strip()
    if choice:
        try:
            number = int(choice)
            if 1 <= number <= len(NAME_MODES):
                NAME_MODE = NAME_MODES[number - 1]
            else:
                NAME_MODE = saved_mode if saved_mode in NAME_MODES else NAME_MODES[0]
        except ValueError:
            NAME_MODE = saved_mode if saved_mode in NAME_MODES else NAME_MODES[0]
    else:
        NAME_MODE = saved_mode if saved_mode in NAME_MODES else NAME_MODES[0]

    try:
        delay = float(settings.get("message_delay_seconds", MESSAGE_DELAY_SECONDS))
        MESSAGE_DELAY_SECONDS = max(0.0, min(delay, 30.0))
    except (TypeError, ValueError):
        MESSAGE_DELAY_SECONDS = 2.0

    settings["name_mode"] = NAME_MODE
    settings["message_delay_seconds"] = MESSAGE_DELAY_SECONDS
    with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)
    print(f"Output names: {NAME_MODE}; fixed posting delay: {MESSAGE_DELAY_SECONDS:g}s")


def _slugify_name(name):
    import re
    import unicodedata
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    value = value.lower().replace("♀", "").replace("♂", "")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


def output_name_for_id(pokemon_id):
    """Return only the selected Pokémon name; never a catch command."""
    if not (1 <= pokemon_id <= len(pokemon_names)):
        return "Unknown Pokémon"

    canonical = str(pokemon_names[pokemon_id - 1]).strip()
    record = multilingual_by_id.get(pokemon_id, {})
    names = record.get("names", {}) if isinstance(record, dict) else {}
    english = str(names.get("english") or canonical).strip()
    mode_to_key = {
        "English": "english",
        "German": "german",
        "French": "french",
        "Japanese Romanized": "japanese_romanized",
        "Japanese Script": "japanese",
    }

    if NAME_MODE == "My Alt Names (TXT + Current Rules)":
        slug = str(record.get("slug") or _slugify_name(canonical))
        custom = alt_names_by_slug.get(slug, {})
        alt = str(custom.get("alt_name", "")).strip() if isinstance(custom, dict) else ""
        # Current naming rules are the fallback when the custom TXT has no value.
        return alt or canonical

    if NAME_MODE == "Current Naming Rules":
        return canonical

    if NAME_MODE == "Alternate English":
        # Use a distinct alternate-English field if supplied by the data; otherwise
        # safely fall back to the canonical English name.
        return str(names.get("english_alternate") or names.get("alternate_english") or english).strip()

    if NAME_MODE == "All Available Languages":
        ordered_keys = ["english", "german", "french", "japanese_romanized", "japanese"]
        available = []
        for key in ordered_keys:
            value = str(names.get(key) or (canonical if key == "english" else "")).strip()
            if value and value not in available:
                available.append(value)
        return " | ".join(available) if available else canonical

    if NAME_MODE == "Random Language":
        import random
        choices = [str(names.get(key) or "").strip() for key in
                   ("english", "german", "french", "japanese_romanized", "japanese")]
        choices = [value for value in choices if value]
        return random.choice(choices) if choices else canonical

def get_first_name(names, keys):
    """Return the first non-empty name from the available field names."""
    for key in keys:
        value = names.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


if NAME_MODE == "Japanese Romanized":
    return (
        get_first_name(names, [
            "japanese_romanized",
            "ja_rm",
            "ja_t",
            "romaji",
            "japanese_romaji",
        ])
        or canonical
    )

if NAME_MODE == "Japanese Script":
    return (
        get_first_name(names, [
            "japanese",
            "ja",
            "ja_jp",
            "japanese_script",
        ])
        or canonical
    )

key = mode_to_key.get(NAME_MODE)
if key:
    return str(names.get(key) or english or canonical).strip()

return canonical



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
            text_values[index].item(),
            pokemon_id
        ))

    results.sort(key=lambda result: result[1], reverse=True)

    return results[:5]

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

    for rank, (name, score, image_score, text_score, pokemon_id) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank}. {name} "
            f"| combined: {score:.4f} "
            f"| image: {image_score:.3f} "
            f"| text: {text_score:.3f}"
        )

    best_id = results[0][4]
    best_name = output_name_for_id(best_id)

    print()
    print(f"Best prediction ID: {best_id}")
    print(f"Name to post: {best_name}")
    print(f"Waiting {MESSAGE_DELAY_SECONDS:g} seconds before posting the name...")

    try:
        if MESSAGE_DELAY_SECONDS:
            await asyncio.sleep(MESSAGE_DELAY_SECONDS)
        # Post only the selected Pokémon name to the same channel as the spawn.
        # The message is intentionally left in the channel; no catch command is
        # copied, prepared, or sent, and no message is deleted.
        await asyncio.sleep(random.uniform(1.2, 3.6))
        sent = await message.channel.send(f"89%say {best_name}")

        # Delete the bot's message after 2 seconds
        await asyncio.sleep(2)
        await sent.delete()

        print(f"Posted name in #{channel_name}: {best_name}")
    except discord.Forbidden:
        print("Cannot post in this channel: the bot needs Send Messages permission.")
    except Exception as error:
        print(f"Could not post Pokémon name: {error}")

    print("No catch command was copied or sent.")
    print("=" * 45)

    try:
        notification.notify(
            title=f"Pokémon detected: {best_name}",
            message=f"Posted name in #{channel_name}",
            timeout=8
        )
    except Exception as error:
        print(f"Notification error: {error}")


# ==========================================
# START
# ==========================================

if __name__ == "__main__":
    load_name_data()
    choose_name_mode()
    load_recognition_model()
    client.run(TOKEN)