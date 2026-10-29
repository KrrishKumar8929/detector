import time
import pyautogui
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
area = (337, 96, 734, 924)
img = pyautogui.screenshot(region=area).convert("RGB")

for scale in (1.0, 0.75, 0.5):
    test = img if scale == 1.0 else img.resize((int(img.width * scale), int(img.height * scale)))
    start = time.perf_counter()
    data = pytesseract.image_to_data(test, output_type=pytesseract.Output.DICT, config="--psm 11")
    elapsed = time.perf_counter() - start
    words = [w.strip() for w in data["text"] if w.strip()]
    print(f"Scale: {scale:.2f} | OCR: {elapsed:.3f}s | Words: {len(words)}")
    print("Full spawn detected:", all(any(w.lower().strip("!.,:;") == key for w in words) for key in ("wild", "pokemon", "appeared")))
