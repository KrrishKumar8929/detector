from pathlib import Path

p = Path("desktop_detector_speedtest.py")
s = p.read_text(encoding="utf-8")

old = '''    data = pytesseract.image_to_data(
        screenshot,
        output_type=pytesseract.Output.DICT,
        config="--psm 11"
    )

    words = []

    for i, raw_word in enumerate(data["text"]):
        word = raw_word.strip()

        if not word:
            continue

        words.append({
            "text": word,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i]
        })
'''

new = '''    scale = 0.5
    small = screenshot.resize((
        max(1, int(screenshot.width * scale)),
        max(1, int(screenshot.height * scale))
    ))

    data = pytesseract.image_to_data(
        small,
        output_type=pytesseract.Output.DICT,
        config="--psm 11"
    )

    words = []

    for i, raw_word in enumerate(data["text"]):
        word = raw_word.strip()

        if not word:
            continue

        words.append({
            "text": word,
            "x": int(data["left"][i] / scale),
            "y": int(data["top"][i] / scale),
            "w": int(data["width"][i] / scale),
            "h": int(data["height"][i] / scale)
        })
'''

if old not in s:
    raise SystemExit("STOP: Expected code not found. No changes made.")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Speed-test copy updated successfully. Original detector untouched.")
