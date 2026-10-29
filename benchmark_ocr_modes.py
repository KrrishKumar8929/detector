import time
import pyautogui
import pytesseract
import tkinter as tk

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

root = tk.Tk()
root.attributes("-fullscreen", True)
root.attributes("-alpha", 0.3)
root.attributes("-topmost", True)
root.configure(bg="black")

canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
canvas.pack(fill="both", expand=True)

state = {"x": 0, "y": 0, "rect": None, "box": None}

def press(event):
    state["x"], state["y"] = event.x_root, event.y_root
    state["rect"] = canvas.create_rectangle(
        event.x, event.y, event.x, event.y,
        outline="lime", width=3
    )

def drag(event):
    if state["rect"] is not None:
        canvas.coords(
            state["rect"],
            state["x"] - root.winfo_rootx(),
            state["y"] - root.winfo_rooty(),
            event.x, event.y
        )

def release(event):
    x1, y1 = state["x"], state["y"]
    x2, y2 = event.x_root, event.y_root
    state["box"] = (
        min(x1, x2), min(y1, y2),
        abs(x2 - x1), abs(y2 - y1)
    )
    root.destroy()

canvas.bind("<ButtonPress-1>", press)
canvas.bind("<B1-Motion>", drag)
canvas.bind("<ButtonRelease-1>", release)
root.bind("<Escape>", lambda event: root.destroy())

print("Select your normal Discord detection area.", flush=True)
root.mainloop()

box = state["box"]
if not box or box[2] < 100 or box[3] < 100:
    raise SystemExit("Selection cancelled or too small.")

image = pyautogui.screenshot(region=box).convert("RGB")

for psm in (11, 6, 12, 3):
    config = f"--psm {psm}"
    print(f"\nTesting {config}...", flush=True)

    try:
        start = time.perf_counter()
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config=config
        )
        elapsed = time.perf_counter() - start

        words = [w.strip() for w in data["text"] if w.strip()]
        notification_found = (
            any("wild" in w.lower() for w in words)
            and any("appeared" in w.lower() for w in words)
        )

        print(f"Time: {elapsed:.3f} seconds", flush=True)
        print(f"Notification words found: {notification_found}", flush=True)
        print(f"Detected words: {len(words)}", flush=True)

    except Exception as exc:
        print(f"Error: {exc}", flush=True)
