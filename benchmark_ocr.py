import time
import pyautogui
import pytesseract
import json
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Read the selected-area coordinates from the detector's source
# without importing it or loading the AI model.
source = Path("desktop_detector.py").read_text(encoding="utf-8")

print("Your detector uses a selected area during monitoring.")
print("The coordinates are stored in memory, not saved to this file.")
print()
print("Please use the Select Area button to select your normal area.")
print("For this benchmark, enter its screen coordinates below.")
