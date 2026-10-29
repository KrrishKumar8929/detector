import time
import pygetwindow as gw
import pyautogui

windows = [
w for w in gw.getAllWindows()
if " - discord" in w.title.lower()
and w.width > 100
and w.height > 100
]

print("Actual Discord windows found:", len(windows))

if windows:
w = windows[0]
print("Selected Discord:", w.title)

try:
    if w.isMinimized:
        w.restore()
        time.sleep(1)

    w.activate()
    time.sleep(2)

    screenshot = pyautogui.screenshot()
    screenshot.save(r"C:\PokemonDetector\discord_capture_test.png")

    print("Full-screen screenshot saved successfully.")

except Exception as error:
    print("Error:", error)

else:
print("Discord not found. Make sure Discord Desktop is open.")