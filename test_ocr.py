import time
import pyautogui
import pytesseract

# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

print("You have 5 seconds to open Discord...")
time.sleep(5)

# Capture the screen
screenshot = pyautogui.screenshot()

# Save the screenshot for inspection
screenshot.save("ocr_test_screenshot.png")

# Read text from the screenshot
text = pytesseract.image_to_string(screenshot)

print("\n========== TEXT DETECTED ==========\n")
print(text)
print("\n===================================")
print("Screenshot saved as ocr_test_screenshot.png")
