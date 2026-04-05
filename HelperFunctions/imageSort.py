import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Sorter")
        self.root.geometry("1100x850")
        self.root.configure(bg="white")

        self.source_folder = filedialog.askdirectory(
            title="Select the folder containing your images"
        )
        if not self.source_folder:
            messagebox.showinfo("No folder selected", "Exiting application.")
            self.root.destroy()
            return

        self.categories = {
            "healthy": {
                "folder": "healthy",
                "prefix": "hjr"
            },
            "white rust": {
                "folder": "white rust",
                "prefix": "wjr"
            },
            "leaf spot": {
                "folder": "leaf spot",
                "prefix": "sjr"
            }
        }

        self.supported_exts = {
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"
        }

        for category in self.categories.values():
            os.makedirs(os.path.join(self.source_folder, category["folder"]), exist_ok=True)

        self.counters = self.initialize_counters()

        self.image_files = [
            f for f in os.listdir(self.source_folder)
            if os.path.isfile(os.path.join(self.source_folder, f))
            and os.path.splitext(f.lower())[1] in self.supported_exts
        ]
        self.image_files.sort()

        if not self.image_files:
            messagebox.showinfo(
                "No images found",
                "No supported image files were found in the selected folder."
            )
            self.root.destroy()
            return

        self.index = 0
        self.current_image_path = None
        self.tk_img = None

        self.title_label = tk.Label(
            root,
            text="Image Sorting Tool",
            font=("Arial", 20, "bold"),
            bg="white"
        )
        self.title_label.pack(pady=10)

        self.progress_label = tk.Label(
            root,
            text="",
            font=("Arial", 12),
            bg="white"
        )
        self.progress_label.pack(pady=5)

        self.filename_label = tk.Label(
            root,
            text="",
            font=("Arial", 12),
            bg="white"
        )
        self.filename_label.pack(pady=5)

        self.image_label = tk.Label(root, bg="white")
        self.image_label.pack(expand=True, pady=10)

        self.button_frame = tk.Frame(root, bg="white")
        self.button_frame.pack(pady=15)

        tk.Button(
            self.button_frame,
            text="Healthy",
            font=("Arial", 14),
            width=15,
            command=lambda: self.move_current_image("healthy")
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            self.button_frame,
            text="White Rust",
            font=("Arial", 14),
            width=15,
            command=lambda: self.move_current_image("white rust")
        ).grid(row=0, column=1, padx=10)

        tk.Button(
            self.button_frame,
            text="Leaf Spot",
            font=("Arial", 14),
            width=15,
            command=lambda: self.move_current_image("leaf spot")
        ).grid(row=0, column=2, padx=10)

        self.bottom_frame = tk.Frame(root, bg="white")
        self.bottom_frame.pack(pady=10)

        tk.Button(
            self.bottom_frame,
            text="Skip",
            font=("Arial", 12),
            width=12,
            command=self.skip_image
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            self.bottom_frame,
            text="Quit",
            font=("Arial", 12),
            width=12,
            command=self.root.destroy
        ).grid(row=0, column=1, padx=10)

        self.counter_label = tk.Label(
            root,
            text="",
            font=("Arial", 11),
            bg="white"
        )
        self.counter_label.pack(pady=5)

        self.show_current_image()

    def initialize_counters(self):
        counters = {}

        for category_name, category_info in self.categories.items():
            folder_path = os.path.join(self.source_folder, category_info["folder"])
            prefix = category_info["prefix"]

            max_num = 0
            pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$", re.IGNORECASE)

            for filename in os.listdir(folder_path):
                name_only, _ = os.path.splitext(filename)
                match = pattern.match(name_only)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num

            counters[category_name] = max_num + 1

        return counters

    def resize_image_for_display(self, image, max_width=950, max_height=600):
        width, height = image.size
        scale = min(max_width / width, max_height / height, 1.0)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS)

    def show_current_image(self):
        if self.index >= len(self.image_files):
            messagebox.showinfo("Done", "All images have been processed.")
            self.root.destroy()
            return

        filename = self.image_files[self.index]
        self.current_image_path = os.path.join(self.source_folder, filename)

        try:
            image = Image.open(self.current_image_path)
            image = self.resize_image_for_display(image)
            self.tk_img = ImageTk.PhotoImage(image)

            self.image_label.config(image=self.tk_img)
            self.filename_label.config(text=f"Current file: {filename}")
            self.progress_label.config(
                text=f"Image {self.index + 1} of {len(self.image_files)}"
            )
            self.update_counter_label()

        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{filename}\n\n{e}")
            self.index += 1
            self.show_current_image()

    def update_counter_label(self):
        self.counter_label.config(
            text=(
                f"Next names -> "
                f"Healthy: hjr{self.counters['healthy']}   |   "
                f"White Rust: wjr{self.counters['white rust']}   |   "
                f"Leaf Spot: sjr{self.counters['leaf spot']}"
            )
        )

    def move_current_image(self, category_name):
        category_info = self.categories[category_name]
        folder_name = category_info["folder"]
        prefix = category_info["prefix"]

        _, ext = os.path.splitext(self.current_image_path)
        number = self.counters[category_name]
        new_filename = f"{prefix}{number}{ext.lower()}"

        destination = os.path.join(self.source_folder, folder_name, new_filename)

        while os.path.exists(destination):
            number += 1
            new_filename = f"{prefix}{number}{ext.lower()}"
            destination = os.path.join(self.source_folder, folder_name, new_filename)

        try:
            shutil.move(self.current_image_path, destination)
            self.counters[category_name] = number + 1
        except Exception as e:
            messagebox.showerror("Move error", f"Could not move file:\n\n{e}")
            return

        self.index += 1
        self.show_current_image()

    def skip_image(self):
        self.index += 1
        self.show_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSorterApp(root)
    root.mainloop()