from email.mime import image
import os
import json
import threading
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import torch
import open_clip
import mss
import pygetwindow as gw
import time
import pytesseract
import pyautogui
import re

from PIL import Image, ImageTk, ImageChops, ImageStat
from pynput import keyboard

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NAMES_FILE = os.path.join(BASE_DIR, "pokemon_names.json")
IMAGES_DIR = os.path.join(BASE_DIR, "pokemon_images")
FEATURES_FILE = os.path.join(
    BASE_DIR, "pokemon_image_features.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

TOP_RESULTS = 5


# =========================================================
# GLOBAL VARIABLES
# =========================================================

model = None
preprocess = None
tokenizer = None

pokemon_names = []
text_features = None
image_features = None
image_ids = []

root = None
status_label = None
results_frame = None
results_container = None

overlay = None
canvas = None
screen_image = None
canvas_image = None
selection_rectangle = None

start_x = 0
start_y = 0

hotkey_listener = None
busy = False


# =========================================================
# AUTOMATIC POKEMON MONITORING
# =========================================================

monitor_region = None
monitor_running = False

selection_mode = "manual"

monitor_previous_image = None
monitor_candidate_image = None
monitor_stable_count = 0

last_processed_spawn = None

MONITOR_INTERVAL = 1000
STABLE_FRAMES_REQUIRED = 2


# =========================================================
# LOAD DATABASE AND AI MODEL
# =========================================================

def load_database():
    global pokemon_names

    if not os.path.exists(NAMES_FILE):
        raise FileNotFoundError(
            "pokemon_names.json was not found."
        )

    with open(NAMES_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        pokemon_names = list(data.values())
    else:
        pokemon_names = data

    pokemon_names = [
        str(name).replace("-", " ").strip()
        for name in pokemon_names
    ]


def load_ai():
    global model, preprocess, tokenizer
    global text_features, image_features, image_ids

    print("Loading Pokémon database...")
    load_database()

    print("Loading AI model...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    # -----------------------------------------------------
    # Prepare text features
    # -----------------------------------------------------

    print("Preparing Pokémon name recognition...")

    prompts = [
        f"a picture of the Pokémon {name}"
        for name in pokemon_names
    ]

    feature_batches = []

    with torch.no_grad():
        for start in range(0, len(prompts), 64):
            batch = prompts[start:start + 64]

            tokens = tokenizer(batch).to(DEVICE)

            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)

            feature_batches.append(features.cpu())

    text_features = torch.cat(feature_batches, dim=0)

    # -----------------------------------------------------
    # Load saved reference-image features
    # -----------------------------------------------------

    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(
            "pokemon_image_features.pt was not found.\n\n"
            "Run pokemon_image_matcher.py first."
        )

    print("Loading reference-image features...")

    cache = torch.load(
        FEATURES_FILE,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(cache, dict):
        raise RuntimeError(
            "The old Pokémon image cache must be rebuilt. "
            "Delete pokemon_image_features.pt and run "
            "pokemon_image_matcher.py again."
        )

    image_features = cache["features"]
    image_ids = cache["image_ids"].tolist()

    print("AI is ready!")

# =========================================================
# CREATE MAIN WINDOW
# =========================================================

def create_main_window():
    global root, status_label, results_frame, results_container

    root = tk.Tk()

    root.title("Pokémon Detector Pro")
    root.geometry("410x510+100+100")
    root.attributes("-topmost", True)

    root.configure(bg="#20242c")

    title = tk.Label(
        root,
        text="POKÉMON DETECTOR",
        font=("Segoe UI", 17, "bold"),
        fg="white",
        bg="#20242c"
    )
    title.pack(pady=(15, 5))

    subtitle = tk.Label(
        root,
        text="AI recognition + reference-image matching",
        font=("Segoe UI", 9),
        fg="#c8d0dc",
        bg="#20242c"
    )
    subtitle.pack(pady=(0, 12))

    select_button = tk.Button(
        root,
        text="SELECT POKÉMON IMAGE  (F8)",
        font=("Segoe UI", 11, "bold"),
        command=start_selection,
        bg="#3b82f6",
        fg="white",
        activebackground="#2563eb",
        activeforeground="white",
        relief="flat",
        padx=12,
        pady=10
    )

    select_button.pack(fill="x", padx=18, pady=5)

    # STOP MONITORING BUTTON
    stop_monitor_button = tk.Button(
        root,
        text="STOP MONITORING",
        font=("Segoe UI", 10, "bold"),
        command=stop_monitoring,
        bg="#7f1d1d",
        fg="white",
        activebackground="#991b1b",
        activeforeground="white",
        relief="flat",
        padx=12,
        pady=8
    )
    stop_monitor_button.pack(fill="x", padx=18, pady=5)

    status_label = tk.Label(
        root,
        text="Loading AI...",
        font=("Segoe UI", 9),
        fg="#fbbf24",
        bg="#20242c",
        wraplength=370
    )


    status_label = tk.Label(
        root,
        text="Loading AI...",
        font=("Segoe UI", 9),
        fg="#fbbf24",
        bg="#20242c",
        wraplength=370
    )
    status_label.pack(pady=10)

    separator = ttk.Separator(root, orient="horizontal")
    separator.pack(fill="x", padx=15, pady=5)

    results_title = tk.Label(
        root,
        text="TOP PREDICTIONS",
        font=("Segoe UI", 11, "bold"),
        fg="white",
        bg="#20242c"
    )
    results_title.pack(pady=5)

    results_frame = tk.Frame(root, bg="#20242c")
    results_frame.pack(fill="both", expand=True, padx=15, pady=5)

    results_container = results_frame

    instructions = tk.Label(
        root,
        text="F8 = Select image   |   Esc = Cancel selection",
        font=("Segoe UI", 8),
        fg="#aeb8c7",
        bg="#20242c"
    )
    instructions.pack(pady=(3, 10))

    root.protocol("WM_DELETE_WINDOW", close_program)

    # Start global keyboard listener
    start_hotkey_listener()

    root.after(1000, finish_loading)

    return root


# =========================================================
# LOAD MODEL WITHOUT FREEZING THE WINDOW
# =========================================================

def finish_loading():
    status_label.config(
        text="AI is loading in the background...",
        fg="#fbbf24"
    )

    def worker():
        try:
            load_ai()

            root.after(
                0,
                lambda: status_label.config(
                    text="Ready! Press F8 or click the button.",
                    fg="#4ade80"
                )
            )

        except Exception as error:
            message = str(error)

            root.after(
                0,
                lambda: status_label.config(
                    text="Loading failed. Check the terminal.",
                    fg="#f87171"
                )
            )

            print("\nERROR WHILE LOADING AI:")
            print(message)

            root.after(
                0,
                lambda: messagebox.showerror(
                    "AI Loading Error",
                    message
                )
            )

    threading.Thread(target=worker, daemon=True).start()


# =========================================================
# GLOBAL F8 SHORTCUT
# =========================================================

def start_hotkey_listener():
    global hotkey_listener

    def on_press(key):
        try:
            if key == keyboard.Key.f8:
                if root:
                    root.after(0, start_selection)

            elif key == keyboard.Key.f9:
                if root:
                    root.after(0, start_monitor_selection)

        except Exception as error:
            print("HOTKEY ERROR:", error)

    hotkey_listener = keyboard.Listener(
        on_press=on_press
    )

    hotkey_listener.daemon = True
    hotkey_listener.start()


# =========================================================
# SCREENSHOT SELECTION
# =========================================================

def start_selection():
    global overlay, canvas, screen_image, canvas_image
    global selection_rectangle, start_x, start_y

    if busy:
        return

    if model is None:
        status_label.config(
            text="Please wait for the AI to finish loading.",
            fg="#fbbf24"
        )
        return

    try:
        root.withdraw()
        root.update()

        with mss.mss() as screenshot_tool:
            monitor = screenshot_tool.monitors[1]
            screenshot = screenshot_tool.grab(monitor)

            screen_image = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb
            )

        # Darken the screenshot so the selection area is visible.
        display_image = screen_image.copy()

        dark_overlay = Image.new(
            "RGB",
            display_image.size,
            (0, 0, 0)
        )

        display_image = Image.blend(
            display_image,
            dark_overlay,
            0.30
        )

        overlay = tk.Toplevel(root)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-topmost", True)
        overlay.overrideredirect(True)

        canvas = tk.Canvas(
            overlay,
            cursor="crosshair",
            highlightthickness=0
        )
        canvas.pack(fill="both", expand=True)

        canvas_image = ImageTk.PhotoImage(display_image)

        canvas.create_image(
            0,
            0,
            image=canvas_image,
            anchor="nw"
        )

        canvas.create_text(
            25,
            25,
            text="Drag around the Pokémon image. Press ESC to cancel.",
            anchor="nw",
            fill="white",
            font=("Segoe UI", 16, "bold")
        )

        selection_rectangle = None

        canvas.bind("<ButtonPress-1>", selection_start)
        canvas.bind("<B1-Motion>", selection_move)
        canvas.bind("<ButtonRelease-1>", selection_finish)

        overlay.bind("<Escape>", cancel_selection)
        overlay.focus_force()

    except Exception as error:
        if overlay:
            overlay.destroy()

        overlay = None

        root.deiconify()

        messagebox.showerror(
            "Selection Error",
            str(error)
        )


def selection_start(event):
    global start_x, start_y, selection_rectangle

    start_x = event.x
    start_y = event.y

    if selection_rectangle is not None:
        canvas.delete(selection_rectangle)

    selection_rectangle = canvas.create_rectangle(
        start_x,
        start_y,
        start_x,
        start_y,
        outline="#00ff66",
        width=3
    )


def selection_move(event):
    if selection_rectangle is not None:
        canvas.coords(
            selection_rectangle,
            start_x,
            start_y,
            event.x,
            event.y
        )


def selection_finish(event):
    global overlay, screen_image
    global selection_mode
    global monitor_region, monitor_running
    global monitor_previous_image
    global monitor_candidate_image
    global monitor_stable_count

    x1 = min(start_x, event.x)
    y1 = min(start_y, event.y)
    x2 = max(start_x, event.x)
    y2 = max(start_y, event.y)

    if x2 - x1 < 20 or y2 - y1 < 20:
        cancel_selection()
        return

    selected_image = screen_image.crop(
        (x1, y1, x2, y2)
    )

    image_path = os.path.join(
        BASE_DIR,
        "selected_pokemon.png"
    )

    selected_image.save(image_path)

    overlay.destroy()
    overlay = None

    root.deiconify()
    root.lift()

    # ---------------------------------------------
    # Automatic monitoring mode
    # ---------------------------------------------

    if selection_mode == "monitor":

        monitor_region = {
            "left": x1,
            "top": y1,
            "width": x2 - x1,
            "height": y2 - y1
        }

        monitor_previous_image = selected_image.copy()
        monitor_candidate_image = None
        monitor_stable_count = 0

        monitor_running = True
        selection_mode = "manual"

        status_label.config(
            text="Monitoring Discord for new images...",
            fg="#4ade80"
        )

        root.after(
            MONITOR_INTERVAL,
            monitor_screen
        )

        return

    # ---------------------------------------------
    # Normal F8 selection mode
    # ---------------------------------------------

    selection_mode = "manual"

    status_label.config(
        text="Image selected! Analyzing...",
        fg="#fbbf24"
    )

    analyze_selected_image(image_path)


def cancel_selection(event=None):
    global overlay

    if overlay is not None:
        overlay.destroy()
        overlay = None

    root.deiconify()
    root.lift()

    status_label.config(
        text="Selection cancelled. Press F8 to try again.",
        fg="#c8d0dc"
    )

# =========================================================
# AUTOMATIC POKEMON MONITORING
# =========================================================

def start_monitor_selection():
    global selection_mode

    if monitor_running:
        status_label.config(
            text="Monitoring is already running.",
            fg="#4ade80"
        )
        return

    selection_mode = "monitor"

    status_label.config(
        text="Select the Discord area to monitor.",
        fg="#fbbf24"
    )

    start_selection()


def stop_monitoring():
    global monitor_running
    global monitor_region
    global monitor_previous_image
    global monitor_candidate_image
    global monitor_stable_count

    monitor_running = False
    monitor_region = None

    monitor_previous_image = None
    monitor_candidate_image = None
    monitor_stable_count = 0

    status_label.config(
        text="Automatic monitoring stopped.",
        fg="#c8d0dc"
    )


def compare_images(image1, image2):
    """
    Compare two images after resizing them.
    Returns the average pixel difference.
    """

    size = (64, 64)

    image1 = image1.convert("L").resize(size)
    image2 = image2.convert("L").resize(size)

    difference = ImageChops.difference(image1, image2)

    statistics = ImageStat.Stat(difference)

    return statistics.mean[0]

def find_poketwo_spawn(screenshot_image):
    """
    Find the lowest visible Pokétwo spawn notification.
    Returns its bounding box, or None if no spawn is found.
    """

    import pytesseract
    import re

    image = screenshot_image.convert("RGB")

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 11"
    )

    spawn_patterns = [
        r"a wild pok[eé]mon has appeared",
        r"wild .* fled.*a new wild pok[eé]mon has appeared"
    ]

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

    matches = []

    for words in lines.values():
        text = " ".join(item["text"] for item in words)
        normalized = re.sub(r"\s+", " ", text.lower())

        if any(
            re.search(pattern, normalized)
            for pattern in spawn_patterns
        ):
            box = (
                min(item["left"] for item in words),
                min(item["top"] for item in words),
                max(item["left"] + item["width"] for item in words),
                max(item["top"] + item["height"] for item in words)
            )

            matches.append(box)

    if not matches:
        return None

    # Select the notification lowest on the screen.
    return max(matches, key=lambda box: box[3])

def crop_pokemon_below_spawn(screenshot_image, notification_box):
    """
    Crop the expected Pokémon image area beneath the spawn notification.

    Returns a PIL image, or None if the crop is invalid.
    """

    left, top, right, bottom = notification_box

    screen_width, screen_height = screenshot_image.size

    # Include some space around the notification horizontally.
    crop_left = max(0, left - 10)
    crop_right = min(screen_width, right + 350)

    # Start beneath the notification.
    crop_top = min(screen_height, bottom + 65)

    # Pokémon images in the tested Discord layout appear below the text.
    crop_bottom = min(screen_height, crop_top + 400)

    if crop_right <= crop_left or crop_bottom <= crop_top:
        return None

    return screenshot_image.crop((
        crop_left,
        crop_top,
        crop_right,
        crop_bottom
    ))


def monitor_screen():
    global monitor_running
    global monitor_previous_image
    global monitor_candidate_image
    global monitor_stable_count
    global last_processed_spawn

    if not monitor_running:
        return

    try:
        # Find the Discord window.
        discord_windows = [
            window for window in gw.getAllWindows()
            if "discord" in window.title.lower()
        ]

        # Pause when Discord is closed or minimized.
        if not discord_windows or discord_windows[0].isMinimized:
            status_label.config(
                text="Discord minimized. Monitoring paused.",
                fg="#fbbf24"
            )

            root.after(MONITOR_INTERVAL, monitor_screen)
            return

        # Capture the current screen.
        screenshot_image = pyautogui.screenshot().convert("RGB")

        # Find the newest visible Pokétwo spawn notification.
        notification_box = find_poketwo_spawn(screenshot_image)

        if notification_box is None:
            status_label.config(
                text="Waiting for a Pokétwo spawn...",
                fg="#60a5fa"
            )

        else:
            # Crop the area below the notification.
            cropped_image = crop_pokemon_below_spawn(
                screenshot_image,
                notification_box
            )

            if cropped_image is not None:
                # Create a fingerprint to avoid analyzing the same
                # unchanged crop repeatedly.
                import hashlib

                fingerprint = hashlib.sha256(
                    cropped_image.tobytes()
                ).hexdigest()

                if fingerprint != last_processed_spawn and not busy:
                    image_path = os.path.join(
                        BASE_DIR,
                        "selected_pokemon.png"
                    )

                    cropped_image.save(image_path)
                    last_processed_spawn = fingerprint

                    analysis_path = os.path.join(
                        BASE_DIR,
                        "pokemon_analysis_snapshot.png"
                    )

                    cropped_image.save(analysis_path)

                    status_label.config(
                        text="Spawn detected! Analyzing Pokémon...",
                        fg="#fbbf24"
                    )

                    print("\nAutomatic detection:")
                    print("Saved image:", image_path)
                    print("Analysis snapshot:", analysis_path)
                    print("Image size:", cropped_image.size)

                    root.after(
                        100,
                        lambda path=analysis_path: analyze_selected_image(path)
                    )
            
    except Exception as error:
        print("MONITORING ERROR:", error)

        status_label.config(
            text="Monitoring error. Check the terminal.",
            fg="#f87171"
        )

    # Continue monitoring while automatic mode is active.
    if monitor_running:
        root.after(MONITOR_INTERVAL, monitor_screen)



# =========================================================
# ANALYZE IMAGE USING BOTH METHODS
# =========================================================

def analyze_selected_image(image_path):
    global busy

    if busy:
        return

    busy = True

    for widget in results_container.winfo_children():
        widget.destroy()

    status_label.config(
        text="Analyzing image with two AI methods...",
        fg="#fbbf24"
    )

    def worker():
        try:
            image = Image.open(image_path).convert("RGB")
            print("\n" + "=" * 50)
            print("IMAGE BEING ANALYZED:", image_path)
            print("IMAGE SIZE:", image.size)
            print("IMAGE FILE:", os.path.getmtime(image_path))
            print("=" * 50)
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)

            # ---------------------------------------------
            # IMAGE-ONLY POKEMON RECOGNITION
            # ---------------------------------------------

            # Compare the selected image against reference images.
            with torch.no_grad():
                query_features = model.encode_image(image_input)
                query_features /= query_features.norm(
                    dim=-1,
                    keepdim=True
                )

                image_scores = (
                    query_features.cpu() @ image_features.T
                ).squeeze(0)

            # Rank reference images by visual similarity.
            image_order = torch.argsort(
                image_scores,
                descending=True
            ).tolist()

            # Keep the best score for each Pokémon.
            best_scores = {}

            for position in image_order:
                pokemon_id = image_ids[position]

                if not (1 <= pokemon_id <= len(pokemon_names)):
                    continue

                name = pokemon_names[pokemon_id - 1]
                score = image_scores[position].item()

                if (
                    name not in best_scores
                    or score > best_scores[name]
                ):
                    best_scores[name] = score

            # Display the Pokémon with the highest image similarity.
            ranked_results = sorted(
                best_scores.items(),
                key=lambda item: item[1],
                reverse=True
            )[:TOP_RESULTS]

            print("\nTOP POKEMON SCORES:")
            for name, score in ranked_results:
                print(f"{name}: {score:.4f}")

            root.after(
                0,
                lambda: display_results(ranked_results)
            )

        except Exception as error:
            message = str(error)

            root.after(
                0,
                lambda: status_label.config(
                    text="Recognition failed. Check the terminal.",
                    fg="#f87171"
                )
            )

            print("\nRECOGNITION ERROR:")
            print(message)

        finally:
            root.after(0, finish_analysis)

    threading.Thread(target=worker, daemon=True).start()


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(results):
    for widget in results_container.winfo_children():
        widget.destroy()

    if not results:
        status_label.config(
            text="No predictions were generated.",
            fg="#f87171"
        )
        return
    
    print("\nRESULTS BEING DISPLAYED:")
    for name, score in results:
        print(f"{name}: {score:.4f}")

    for rank, (name, score) in enumerate(results, start=1):
        row = tk.Frame(
            results_container,
            bg="#303743",
            padx=8,
            pady=7
        )

        row.pack(fill="x", pady=3)

        name_label = tk.Label(
            row,
            text=f"{rank}. {name.lower()}",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg="#303743",
            anchor="w"
        )

        name_label.pack(
            side="left",
            fill="x",
            expand=True
        )

        bot2_button = tk.Button(
            row,
            text="BOT 2",
            font=("Segoe UI", 8, "bold"),
            command=lambda pokemon=name: copy_name(pokemon, bot=2),
            bg="#15803d",
            fg="white",
            relief="flat"
        )

        bot2_button.pack(side="right", padx=(4, 0))

        bot1_button = tk.Button(
            row,
            text="BOT 1",
            font=("Segoe UI", 8, "bold"),
            command=lambda pokemon=name: copy_name(pokemon, bot=1),
            bg="#2563eb",
            fg="white",
            relief="flat"
        )

        bot1_button.pack(side="right", padx=(0, 4))

    status_label.config(
        text="Analysis complete! Choose a bot to copy its command.",
        fg="#4ade80"
    )



def copy_name(name, bot=1):
    name = name.lower()

    # Build the command for the selected bot.
    if bot == 1:
        command = f"@Pokétwo#8236 c {name}"
    else:
        command = f"p!c {name}"

    # Copy the complete command.
    root.clipboard_clear()
    root.clipboard_append(command)
    root.update()

    status_label.config(
        text=f"Copied: {command}",
        fg="#4ade80"
    )

    # Return focus to Discord.
    try:
        discord_windows = [
            window for window in gw.getAllWindows()
            if "discord" in window.title.lower()
            and window.visible
        ]

        if discord_windows:
            discord_window = discord_windows[0]
            root.after(300, discord_window.activate)

    except Exception as e:
        print(f"Could not activate Discord: {e}")



def finish_analysis():
    global busy
    busy = False


# =========================================================
# CLOSE PROGRAM
# =========================================================

def close_program():
    global hotkey_listener

    try:
        if hotkey_listener:
            hotkey_listener.stop()
    except Exception:
        pass

    try:
        if overlay:
            overlay.destroy()
    except Exception:
        pass

    root.destroy()


# =========================================================
# START PROGRAM
# =========================================================

if __name__ == "__main__":
    create_main_window()
    root.mainloop()