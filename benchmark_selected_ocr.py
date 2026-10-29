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
            event.x,
            event.y
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

print("Drag around the SAME area you normally select in the detector.")
print("Press Esc to cancel.", flush=True)
root.mainloop()

box = state["box"]
if not box or box[2] < 100 or box[3] < 100:
    raise SystemExit("Selection cancelled or too small.")

print(f"Selected area: {box[2]} x {box[3]}", flush=True)
image = pyautogui.screenshot(region=box).convert("RGB")

pytesseract.image_to_data(
    image, output_type=pytesseract.Output.DICT, config="--psm 11"
)

times = []
for i in range(3):
    start = time.perf_counter()
    pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT, config="--psm 11"
    )
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"OCR run {i+1}: {elapsed:.3f} seconds", flush=True)

print(f"Average OCR time: {sum(times)/len(times):.3f} seconds", flush=True)
