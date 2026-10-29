import time
import pytesseract
import pyautogui

region = (337, 96, 734, 924)
screenshot = pyautogui.screenshot(region=region).convert("RGB")

for mode in (11, 6):
print(f"\nPSM {mode}")
for i in range(3):
start = time.perf_counter()
data = pytesseract.image_to_data(
screenshot,
output_type=pytesseract.Output.DICT,
config=f"--psm {mode}"
)
elapsed = time.perf_counter() - start
count = sum(bool(word.strip()) for word in data["text"])
print(f"Scan {i + 1}: {elapsed:.3f}s, {count} words")
