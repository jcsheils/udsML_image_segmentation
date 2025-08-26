# GENERATED WITH GEMINI

import os
import numpy as np
from PIL import Image, ImageDraw

# --- SOLUTION: Explicitly set the backend ---
# This must be done BEFORE importing pyplot
import matplotlib
matplotlib.use('TkAgg') 
# -------------------------------------------

import matplotlib.pyplot as plt

# --- Configuration ---
ROOT_DIR = "data"
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
SCRIBBLE_DIR = os.path.join(ROOT_DIR, "scribbles")
TRUTH_DIR = os.path.join(ROOT_DIR, "ground_truth")

# --- Dummy Data Generation (for demonstration) ---
def create_dummy_data(num_files=5, size=(256, 256)):
    """Creates the necessary directories and dummy image files if they don't exist."""
    print("Checking for dummy data...")
    if os.path.exists(ROOT_DIR):
        print("Data directory already exists. Skipping creation.")
        return

    print(f"Creating dummy data in '{ROOT_DIR}' directory...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SCRIBBLE_DIR, exist_ok=True)
    os.makedirs(TRUTH_DIR, exist_ok=True)

    for i in range(1, num_files + 1):
        base_name = str(i)
        image_filename = f"{base_name}.jpg"
        png_filename = f"{base_name}.png"

        img_array = np.zeros((*size, 3), dtype=np.uint8)
        g, r = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        img_array[..., 0] = r / size[0] * (100 + i*30)
        img_array[..., 1] = g / size[1] * (100 + i*30)
        img_array[..., 2] = 50 + i*10
        img = Image.fromarray(img_array)
        img.save(os.path.join(IMAGE_DIR, image_filename))

        scribble = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(scribble)

        # Draw black lines (will have value 0 when converted to grayscale)
        draw.line((30, 30 + i*20, size[0] - 30, size[1] - (30 + i*20)), fill="black", width=3)
        draw.ellipse((size[0]//2 - i*5, size[1]//2 - i*5, size[0]//2 + i*5, size[1]//2 + i*5), outline="black", width=2)
        
        # Draw a new shape with RGB color (1, 1, 1), which will convert to 1 in grayscale
        # This is the shape that will be colored RED in the visualizer
        draw.rectangle((20, size[1]-40, size[0]-20, size[1]-25), fill=(1, 1, 1))

        scribble.save(os.path.join(SCRIBBLE_DIR, png_filename))

        truth_array = np.zeros((*size, 3), dtype=np.uint8)
        truth_array[:, :] = [20*i, 128 - 10*i, 200 - 25*i]
        truth = Image.fromarray(truth_array)
        truth.save(os.path.join(TRUTH_DIR, png_filename))

    print(f"Created {num_files} sets of dummy images (.jpg for images, .png for others).")

# --- The Visualizer Class ---
class ImageVisualizer:
    def __init__(self, image_dir, scribble_dir, truth_dir):
        self.image_dir = image_dir
        self.scribble_dir = scribble_dir
        self.truth_dir = truth_dir

        try:
            self.files = sorted([f for f in os.listdir(self.scribble_dir) if f.endswith('.png')])
            if not self.files:
                raise FileNotFoundError("No .png files found.")
        except FileNotFoundError:
            print(f"Error: Directory '{self.scribble_dir}' not found or is empty.")
            print("Please create the directories or run the script again to generate dummy data.")
            exit()

        self.current_index = 0
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print("\n--- Visualizer Controls ---")
        print("Right Arrow / 'n' -> Next Image")
        print("Left Arrow  / 'p' -> Previous Image")
        print("-------------------------\n")

        self.update_display()
        
    def on_key(self, event):
        if event.key in ['right', 'n']:
            self.current_index = (self.current_index + 1) % len(self.files)
        elif event.key in ['left', 'p']:
            self.current_index = (self.current_index - 1 + len(self.files)) % len(self.files)
        else:
            return
        self.update_display()

    def update_display(self):
        png_filename = self.files[self.current_index]
        base_name, _ = os.path.splitext(png_filename)
        jpg_filename = f"{base_name}.jpg"
        img_path = os.path.join(self.image_dir, jpg_filename)
        scribble_path = os.path.join(self.scribble_dir, png_filename)
        truth_path = os.path.join(self.truth_dir, png_filename)

        try:
            base_image = Image.open(img_path).convert("RGB")
            scribble_image = Image.open(scribble_path).convert("RGB")
            truth_image = Image.open(truth_path)


            # --- New Overlay Logic using NumPy ---
            # Convert images to NumPy arrays for efficient pixel manipulation.
            overlaid_np = np.array(base_image.copy())
            # Convert scribble to a single channel (Luminance) to easily check pixel values.
            scribble_np_gray = np.array(scribble_image.convert('L'))

            # Define the colors for the overlay
            black = [0, 0, 0]
            red = [255, 0, 0]

            # Use boolean indexing to find where the scribble pixels are 0 or 1.
            # Where the scribble is 0 (black lines), set the overlay pixels to black.
            overlaid_np[scribble_np_gray == 0] = black
            
            # Where the scribble is 1, set the overlay pixels to red.
            overlaid_np[scribble_np_gray == 1] = red
            
            # Convert the modified NumPy array back to a PIL Image for display.
            overlaid_image = Image.fromarray(overlaid_np)  

            self.ax1.clear(); self.ax2.clear()
            self.ax1.imshow(overlaid_image)
            self.ax1.set_title("Image + Scribble Overlay"); self.ax1.axis('off')
            self.ax2.imshow(truth_image)
            self.ax2.set_title("Ground Truth"); self.ax2.axis('off')

            self.fig.suptitle(f"File: {base_name} (.jpg/.png) ({self.current_index + 1}/{len(self.files)})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.fig.canvas.draw()

        except FileNotFoundError as e:
            print(f"\nError: Could not find a corresponding file.")
            print(f"Attempted to load:\n - Image: {img_path}\n - Scribble: {scribble_path}\n - Ground Truth: {truth_path}")
            print(f"Original error: {e}")
            self.ax1.clear(); self.ax2.clear()
            self.ax1.text(0.5, 0.5, f"File not found:\n{os.path.basename(e.filename)}", ha='center', va='center', color='red', wrap=True)
            self.ax1.axis('off'); self.ax2.axis('off')
            self.fig.canvas.draw()

    def run(self):
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    create_dummy_data()
    visualizer = ImageVisualizer(IMAGE_DIR, SCRIBBLE_DIR, TRUTH_DIR)
    visualizer.run()