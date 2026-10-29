import tkinter as tk
from tkinter import ttk
import mss
from PIL import Image, ImageTk


class PokemonDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokémon Detector")
        self.root.geometry("420x350+100+100")
        self.root.attributes("-topmost", True)

        # Window title
        title = ttk.Label(
            root,
            text="Pokémon Detector",
            font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        # Instructions
        instructions = ttk.Label(
            root,
            text="Click the button to capture your screen."
        )
        instructions.pack(pady=5)

        # Screenshot display
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=5)

        # Status message
        self.status_label = ttk.Label(root, text="Ready")
        self.status_label.pack(pady=5)

        # Screenshot button
        self.capture_button = ttk.Button(
            root,
            text="Capture Screen",
            command=self.capture_screen
        )
        self.capture_button.pack(pady=10)

    def capture_screen(self):
        try:
            with mss.mss() as screen:
                screenshot = screen.grab(screen.monitors[1])

                image = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.rgb
                )

                # Resize image to fit the window
                image.thumbnail((380, 200))

                self.photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=self.photo)

                self.status_label.configure(
                    text="Screenshot captured successfully!"
                )

        except Exception as error:
            self.status_label.configure(
                text=f"Error: {error}"
            )


if __name__ == "__main__":
    window = tk.Tk()
    app = PokemonDetector(window)
    window.mainloop()