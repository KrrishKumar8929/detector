import os
import re
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import torch
import open_clip
import pyautogui
import pygetwindow as gw
import pytesseract

from PIL import Image, ImageTk, ImageDraw



# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\PokemonDetector"

DATABASE_FILE = os.path.join(
    BASE_DIR, "poketwo_full_reference_features.pt"
)

CATCH_NAME_MAP_FILE = os.path.join(
    BASE_DIR, "pokemon_catch_name_map.json"
)

METADATA_FILE = os.path.join(
    BASE_DIR, "poketwo_full_feature_metadata.json"
)

CROP_FILE = os.path.join(
    BASE_DIR, "desktop_spawn_crop.png"
)

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BOT1_PREFIX = "@Pokétwo#8236 c "
BOT2_PREFIX = "p!c "

SCAN_INTERVAL = 0.1

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# GLOBAL STATE
# ============================================================

model = None
preprocess = None
reference_features = None
reference_ids = None
reference_names = None
id_to_name = {}

monitoring = False
worker = None
last_spawn_key = None
current_prediction = None
latest_photo = None
selected_area = None


# ============================================================
# CATCH-NAME MAP
catch_name_map = {}


def load_catch_name_map():
    global catch_name_map

    with open(CATCH_NAME_MAP_FILE, "r", encoding="utf-8") as f:
        raw_map = json.load(f)

    catch_name_map = {
        str(folder).strip().lower(): str(
            entry.get("catch_name", folder)
        ).strip().lower()
        for folder, entry in raw_map.items()
        if isinstance(entry, dict)
    }


def get_catch_name(reference_folder_name):
    key = str(reference_folder_name).strip().lower()
    return catch_name_map.get(key, key)


# LOAD AI DATABASE
# ============================================================

def load_ai():
    global model, preprocess
    global reference_features, reference_ids
    global reference_names, id_to_name

    status_var.set("Loading catch-name map...")
    root.update_idletasks()
    load_catch_name_map()


    status_var.set("Loading Pokémon metadata...")
    root.update_idletasks()

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    id_to_name = {
        int(item["pokemon_id"]): item["canonical_name"].lower()
        for item in metadata.values()
    }

    status_var.set("Loading reference-image database...")
    root.update_idletasks()

    database = torch.load(
        DATABASE_FILE,
        map_location="cpu",
        weights_only=False
    )

    reference_features = database["features"].float()
    reference_ids = database["image_ids"].long()

    # Preserve the Pokémon form name from each reference image's folder.
    reference_names = [
        os.path.basename(os.path.dirname(path))
        .replace("_", " ")
        .strip()
        .lower()
        for path in database["image_paths"]
    ]

    status_var.set("Loading CLIP model...")
    root.update_idletasks()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    model = model.to(device)
    model.eval()

    # Normalize once, instead of every scan.
    reference_features = reference_features / (
        reference_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    )

    status_var.set(
        f"Ready | {len(reference_features):,} reference images | {device.upper()}"
    )

    start_button.config(state="normal")


# ============================================================
# CUSTOM SCREEN AREA SELECTION
# ============================================================

def select_detection_area():
    global selected_area

    if monitoring:
        messagebox.showinfo(
            "Stop Monitoring First",
            "Please stop monitoring before selecting a new area."
        )
        return

    # Temporarily hide the detector so it is not included
    # in the selected screen area.
    root.withdraw()
    time.sleep(0.3)

    screen_width, screen_height = pyautogui.size()

    overlay = tk.Toplevel()
    overlay.overrideredirect(True)
    overlay.attributes("-topmost", True)
    overlay.geometry(
        f"{screen_width}x{screen_height}+0+0"
    )
    overlay.configure(bg="black")
    overlay.attributes("-alpha", 0.25)

    canvas = tk.Canvas(
        overlay,
        cursor="cross",
        bg="black",
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    start_x = 0
    start_y = 0
    rectangle = None

    instruction = canvas.create_text(
        screen_width // 2,
        35,
        text="Drag to select the Pokémon spawn area. Press Esc to cancel.",
        fill="white",
        font=("Segoe UI", 16, "bold")
    )

    def on_press(event):
        nonlocal start_x, start_y, rectangle

        start_x = event.x
        start_y = event.y

        if rectangle is not None:
            canvas.delete(rectangle)

        rectangle = canvas.create_rectangle(
            start_x,
            start_y,
            start_x,
            start_y,
            outline="lime",
            width=3
        )

    def on_drag(event):
        if rectangle is not None:
            canvas.coords(
                rectangle,
                start_x,
                start_y,
                event.x,
                event.y
            )

    def on_release(event):
        global selected_area

        left = min(start_x, event.x)
        top = min(start_y, event.y)
        right = max(start_x, event.x)
        bottom = max(start_y, event.y)

        if right - left < 100 or bottom - top < 100:
            overlay.destroy()
            root.deiconify()
            status_var.set(
                "Area too small. Please select a larger area."
            )
            return

        selected_area = (
            int(left),
            int(top),
            int(right),
            int(bottom)
        )

        overlay.destroy()
        root.deiconify()

        status_var.set(
            f"Detection area saved: {selected_area}"
        )

    def cancel_selection(event=None):
        overlay.destroy()
        root.deiconify()
        status_var.set("Area selection cancelled.")

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    overlay.bind("<Escape>", cancel_selection)

    canvas.focus_set()
    overlay.grab_set()
    root.wait_window(overlay)



# ============================================================
# DISCORD WINDOW
# ============================================================

def get_discord_window():
    windows = [
        w for w in gw.getAllWindows()
        if " - discord" in w.title.lower()
        and w.width > 100
        and w.height > 100
        and not w.isMinimized
    ]

    return windows[0] if windows else None


# ============================================================
# SCREEN OCR AND SPAWN DETECTION
# ============================================================

def read_screen():
    """Capture the selected screen area or the Discord window."""

    if selected_area is not None:
        left, top, right, bottom = selected_area
    else:
        window = get_discord_window()

        if window is None:
            return None, []

        screen_width, screen_height = pyautogui.size()

        left = max(0, window.left)
        top = max(0, window.top)
        right = min(screen_width, window.left + window.width)
        bottom = min(screen_height, window.top + window.height)

    screen_width, screen_height = pyautogui.size()

    left = max(0, int(left))
    top = max(0, int(top))
    right = min(screen_width, int(right))
    bottom = min(screen_height, int(bottom))

    if right <= left or bottom <= top:
        return None, []

    screenshot = pyautogui.screenshot(
        region=(left, top, right - left, bottom - top)
    ).convert("RGB")

    data = pytesseract.image_to_data(
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

    return screenshot, words




def locate_spawn(screenshot, words):
    """
    Locate the Poketwo spawn notification and crop the artwork
    directly below its text.sss
    """

    if screenshot is None:
        return None, None

    for word in words:
        normalized = re.sub(
            r"[^a-z]",
            "",
            word["text"].lower()
        )

        if normalized not in ("appeared", "appeare"):
            continue

        x = word["x"]
        y = word["y"]

        # Find words near the notification line.
        line_words = [
            item for item in words
            if abs(item["y"] - y) < 35
            and abs(item["x"] - x) < 1000
        ]

        line_text = " ".join(
            item["text"].lower()
            for item in line_words
        )

        # Confirm this is a Pokémon spawn notification.
        if "wild" not in line_text:
            continue

        if "pokemon" not in line_text and "pokémon" not in line_text:
            continue

        # Crop below the notification text.
        # These offsets can be adjusted after testing.
        left = max(0, x - 350)
        top = min(screenshot.height, y + 65)
        right = min(screenshot.width, x + 450)
        bottom = min(screenshot.height, top + 380)

        if right <= left or bottom <= top:
            continue

        crop = screenshot.crop((left, top, right, bottom))

        return crop, (left, top, right, bottom)

    return None, None


def wait_for_clear_spawn(initial_crop, box=None):
    """
    Wait briefly for Discord's Pokémon artwork to load, then
    recapture the known crop without performing OCR a second time.
    """

    time.sleep(0.15)

    if not monitoring or box is None:
        return initial_crop

    if selected_area is not None:
        origin_x, origin_y = selected_area[0], selected_area[1]
    else:
        window = get_discord_window()
        if window is None:
            return initial_crop
        origin_x, origin_y = max(0, window.left), max(0, window.top)

    left, top, right, bottom = box
    region = (
        origin_x + left,
        origin_y + top,
        right - left,
        bottom - top
    )

    try:
        screenshot = pyautogui.screenshot(region=region).convert("RGB")
        if screenshot.width > 0 and screenshot.height > 0:
            return screenshot
    except Exception as exc:
        print(f"Updated spawn capture failed: {exc}")

    return initial_crop


# ============================================================
# IMAGE RECOGNITION
# ============================================================

def recognize_image(image):
    image_tensor = preprocess(
        image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        query = model.encode_image(image_tensor)
        query = query / query.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-12)

        scores = (
            query.cpu() @ reference_features.T
        ).squeeze(0)

    # Retrieve the strongest reference-image matches.
    count = min(100, scores.numel())
    top_scores, top_indices = scores.topk(count)

    # Group by folder name, preserving regional forms.
    best_by_name = {}

    for score, index in zip(
        top_scores.tolist(),
        top_indices.tolist()
    ):
        pokemon_name = reference_names[index]

        # Use the same catch name for both Nidoran variants.
        if pokemon_name in ("nidoranf", "nidoranm"):
            pokemon_name = "nidoran"

        if (
            pokemon_name not in best_by_name
            or score > best_by_name[pokemon_name]
        ):
            best_by_name[pokemon_name] = score

    ranked = sorted(
        best_by_name.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]

    return ranked



# ============================================================
# GUI HELPERS
# ============================================================

def update_status(message):
    root.after(0, lambda: status_var.set(message))


def display_crop(image):
    global latest_photo

    def update():
        global latest_photo

        preview = image.copy()
        preview.thumbnail((340, 240))

        latest_photo = ImageTk.PhotoImage(preview)

        preview_label.config(image=latest_photo)

    root.after(0, update)


def display_predictions(predictions):
    def update():
        prediction_list.delete(0, tk.END)

        for name, score in predictions:
            prediction_list.insert(
                tk.END,
                f"{name}  |  {score:.4f}"
            )

        if predictions:
            prediction_list.selection_clear(0, tk.END)
            prediction_list.selection_set(0)
            prediction_list.activate(0)

    root.after(0, update)


def copy_command(prefix):
    selection = prediction_list.curselection()

    if not selection:
        messagebox.showwarning(
            "No Pokémon selected",
            "Select a Pokémon prediction first."
        )
        return

    selected_text = prediction_list.get(selection[0])
    reference_folder_name = selected_text.split("|")[0].strip().lower()
    pokemon_name = get_catch_name(reference_folder_name)

    command = prefix + pokemon_name

    # Copy the catch command to the clipboard.
    root.clipboard_clear()
    root.clipboard_append(command)
    root.update()

    status_var.set(f"Copied command: {command}")

    # Find the Discord window.
    try:
        windows = [
            w for w in gw.getAllWindows()
            if " - discord" in w.title.lower()
            and w.width > 100
            and w.height > 100
        ]

        if not windows:
            status_var.set(
                f"Copied command: {command} | Discord not found"
            )
            return

        discord_window = windows[0]

        # Restore Discord if it is minimized.
        if discord_window.isMinimized:
            discord_window.restore()

        # Bring Discord to the foreground.
        discord_window.activate()

        # Allow the window to become active.
        time.sleep(0.15)

        # Focus the message input near the bottom of Discord.
        click_x = discord_window.left + discord_window.width // 2
        click_y = discord_window.top + discord_window.height - 45

        pyautogui.click(click_x, click_y)

        status_var.set(
            f"Copied {pokemon_name}. Press Ctrl+V, then Enter."
        )

    except Exception as error:
        status_var.set(
            f"Command copied. Could not focus Discord: {error}"
        )


# ============================================================
# MONITORING LOOP
# ============================================================

def monitor_loop():
    global monitoring, last_spawn_key, current_prediction

    update_status("Monitoring Discord...")

    while monitoring:
        try:
            window = get_discord_window()

            if window is None:
                update_status(
                    "Paused: Discord is minimized or not found."
                )
                time.sleep(SCAN_INTERVAL)
                continue

            screenshot, words = read_screen()
            if screenshot is None:
                time.sleep(SCAN_INTERVAL)
                continue

            crop, box = locate_spawn(screenshot, words)

            if crop is None:
                update_status(
                    f"Watching {window.title}"
                )
                time.sleep(SCAN_INTERVAL)
                continue

            # Use the OCR text and crop position as an initial
            # duplicate-detection key.
            spawn_words = [
                item["text"].lower()
                for item in words
                if "wild" in item["text"].lower()
                or "appeared" in item["text"].lower()
            ]

            spawn_key = (
                window.title,
                tuple(spawn_words),
                box
            )

            if spawn_key == last_spawn_key:
                time.sleep(SCAN_INTERVAL)
                continue

            update_status(
                "Spawn detected. Waiting for Pokémon image..."
            )

            # Wait until Discord has had time to load the artwork.
            crop = wait_for_clear_spawn(crop, box)

            if not monitoring:
                break

            # Mark the spawn as processed after waiting.
            last_spawn_key = spawn_key

            crop.save(CROP_FILE)

            update_status(
                "Image ready. Identifying Pokémon..."
            )

            display_crop(crop)

            # The initial crop may include Discord text.
            # Recognition is shown for testing and is not
            # automatically used to issue a catch command.
            predictions = recognize_image(crop)

           
            current_prediction = (
                predictions[0][0] if predictions else None
            )

            display_predictions(predictions)

            # Automatically copy the first prediction using BOT 1.
            if current_prediction:
                catch_name = get_catch_name(current_prediction)
                command = "@Pokétwo#8236 c " + catch_name
                root.after(
                    0,
                    lambda cmd=command: (
                        root.clipboard_clear(),
                        root.clipboard_append(cmd),
                        root.update_idletasks()
                    )
                )

                update_status(
                    f"Copied first prediction: {command}"
                )
            else:
                update_status("No prediction available.")


        except Exception as error:
            update_status(f"Monitoring error: {error}")

        time.sleep(SCAN_INTERVAL)

    update_status("Monitoring stopped.")


def start_monitoring():
    global monitoring, worker, last_spawn_key

    if monitoring:
        return

    if model is None:
        messagebox.showerror(
            "AI not ready",
            "Wait until the AI finishes loading."
        )
        return

    last_spawn_key = None
    monitoring = True

    start_button.config(state="disabled")
    stop_button.config(state="normal")

    worker = threading.Thread(
        target=monitor_loop,
        daemon=True
    )

    worker.start()


def stop_monitoring():
    global monitoring

    monitoring = False

    start_button.config(state="normal")
    stop_button.config(state="disabled")


def close_app():
    global monitoring

    monitoring = False
    root.destroy()


# ============================================================
# USER INTERFACE
# ============================================================

root = tk.Tk()

root.title("Pokémon Desktop Detector")
root.geometry("470x700")
root.minsize(450, 650)
root.attributes("-topmost", False)

heading = ttk.Label(
    root,
    text="Pokémon Desktop Detector",
    font=("Segoe UI", 16, "bold")
)

heading.pack(pady=(12, 5))

status_var = tk.StringVar(value="Starting...")

status_label = ttk.Label(
    root,
    textvariable=status_var,
    wraplength=430
)

status_label.pack(pady=5)

preview_label = ttk.Label(
    root,
    text="Spawn image preview will appear here."
)

preview_label.pack(pady=8)

ttk.Label(
    root,
    text="Top 5 predictions"
).pack(pady=(5, 0))

prediction_list = tk.Listbox(
    root,
    height=7,
    font=("Consolas", 11),
    exportselection=False
)

prediction_list.pack(
    fill="x",
    padx=15,
    pady=8
)

button_frame = ttk.Frame(root)
button_frame.pack(pady=5)

ttk.Button(
    button_frame,
    text="Copy BOT 1 Command",
    command=lambda: copy_command(BOT1_PREFIX)
).grid(row=0, column=0, padx=5, pady=5)

ttk.Button(
    button_frame,
    text="Copy BOT 2 Command",
    command=lambda: copy_command(BOT2_PREFIX)
).grid(row=0, column=1, padx=5, pady=5)

area_button = ttk.Button(
    root,
    text="Select Detection Area",
    command=select_detection_area
)
area_button.pack(pady=5)
control_frame = ttk.Frame(root)
control_frame.pack(pady=8)

start_button = ttk.Button(
    control_frame,
    text="Start Monitoring",
    command=start_monitoring,
    state="disabled"
)

start_button.grid(row=0, column=0, padx=5)

stop_button = ttk.Button(
    control_frame,
    text="Stop Monitoring",
    command=stop_monitoring,
    state="disabled"
)

stop_button.grid(row=0, column=1, padx=5)

ttk.Label(
    root,
    text=(
        "Commands are copied only when you click a button.\n"
        "Nothing is sent automatically.\n"
        "Crop testing: desktop_spawn_crop.png"
    ),
    justify="center"
).pack(pady=12)

root.protocol("WM_DELETE_WINDOW", close_app)

# Load the AI in a worker so the window can appear immediately.
def initialize_ai():
    try:
        load_ai()
    except Exception as error:
        root.after(
            0,
            lambda: messagebox.showerror(
                "Initialization error",
                str(error)
            )
        )
        update_status(f"Initialization failed: {error}")


threading.Thread(
    target=initialize_ai,
    daemon=True
).start()

root.mainloop()