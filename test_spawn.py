import os
import re
import pytesseract

from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "ocr_test_screenshot.png")


def find_poketwo_spawn(screenshot_image):
    data = pytesseract.image_to_data(
        screenshot_image.convert("RGB"),
        output_type=pytesseract.Output.DICT,
        config="--psm 11"
    )

    lines = {}

    for i, word in enumerate(data["text"]):
        word = word.strip()

        if not word:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i]
        )

        lines.setdefault(key, []).append({
            "text": word,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i]
        })

    for words in lines.values():
        text = " ".join(item["text"] for item in words)
        text = re.sub(r"\s+", " ", text.lower())

        if (
            re.search(r"a wild pok[eé]mon has appeared", text)
            or re.search(
                r"wild .* fled.*a new wild pok[eé]mon has appeared",
                text
            )
        ):
            box = (
                min(w["left"] for w in words),
                min(w["top"] for w in words),
                max(w["left"] + w["width"] for w in words),
                max(w["top"] + w["height"] for w in words)
            )

            return text, box

    return None, None


if not os.path.exists(IMAGE_PATH):
    print("ERROR: ocr_test_screenshot.png was not found.")
else:
    image = Image.open(IMAGE_PATH)
    text, box = find_poketwo_spawn(image)

    if box:
        print("SUCCESS: Spawn notification detected!")
        print("Recognized text:", text)
        print("Notification coordinates:", box)
    else:
        print("Spawn notification NOT detected.")
        print("We may need to improve the OCR matching.")
