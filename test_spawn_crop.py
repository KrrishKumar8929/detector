import os
import re
import time
import pygetwindow as gw
import pyautogui
import pytesseract
from PIL import Image, ImageDraw

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = r"C:\PokemonDetector"


def find_spawn(image):
    text = pytesseract.image_to_data(
        image.convert("RGB"),
        output_type=pytesseract.Output.DICT,
        config="--psm 11"
    )

    matches = []

    for i, word in enumerate(text["text"]):
        word = word.strip().lower()

        if word in ("appeared", "appeared!", "appeare"):
            matches.append(i)

    for i in matches:
        left = text["left"][i]
        top = text["top"][i]
        right = left + text["width"][i]
        bottom = top + text["height"][i]

        # Search nearby OCR words for the spawn phrase.
        nearby = []

        for j, word in enumerate(text["text"]):
            word = word.strip().lower()

            if not word:
                continue

            x = text["left"][j]
            y = text["top"][j]

            if abs(y - top) < 100 and abs(x - left) < 900:
                nearby.append(word)

        phrase = " ".join(nearby)

        if "wild" in phrase and (
            "pokemon" in phrase or "pokémon" in phrase
        ):
            return (
                max(0, left - 300),
                max(0, top - 100),
                min(image.width, right + 650),
                min(image.height, bottom + 650)
            )

    # Fallback: search the full OCR text.
    full_text = pytesseract.image_to_string(
        image,
        config="--psm 11"
    )

    if re.search(r"wild.*pokemon.*appeared", full_text, re.I | re.S):
        print("Spawn text detected, but its exact position was unclear.")

    return None


windows = [
    w for w in gw.getAllWindows()
    if " - discord" in w.title.lower()
    and w.width > 100
    and w.height > 100
]

print("Discord windows:", [w.title for w in windows])

if not windows:
    print("ERROR: Discord window not found.")
    raise SystemExit(1)

window = windows[0]

if window.isMinimized:
    window.restore()

window.activate()
time.sleep(3)

image = pyautogui.screenshot().convert("RGB")
image.save(os.path.join(BASE_DIR, "discord_capture_test.png"))

print("Screenshot captured:", image.size)

box = find_spawn(image)

if box is None:
    print("Spawn notification not located.")
    print("Saved screenshot: C:\\PokemonDetector\\discord_capture_test.png")
    print("Open that screenshot and check whether the spawn message is visible.")
    raise SystemExit(1)

print("Trial crop coordinates:", box)

drawn = image.copy()
ImageDraw.Draw(drawn).rectangle(box, outline="red", width=4)
drawn.save(os.path.join(BASE_DIR, "spawn_marked.png"))

crop = image.crop(box)
crop.save(os.path.join(BASE_DIR, "spawn_crop_test.png"))

print("Saved: C:\\PokemonDetector\\spawn_marked.png")
print("Saved: C:\\PokemonDetector\\spawn_crop_test.png")
print("Open spawn_crop_test.png and check whether the Pokémon picture is visible.")
