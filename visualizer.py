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
import argparse

# --- The Visualizer Class ---
class ImageVisualizer:
    def __init__(self, image_dir, scribble_dir, truth_dir, prediction_dir):
        self.image_dir = image_dir
        self.scribble_dir = scribble_dir
        self.truth_dir = truth_dir
        self.prediction_dir = prediction_dir

        try:
            self.files = sorted([f for f in os.listdir(self.scribble_dir) if f.endswith('.png')])
            if not self.files:
                raise FileNotFoundError("No .png files found.")
        except FileNotFoundError:
            print(f"Error: Directory '{self.scribble_dir}' not found or is empty.")
            print("Please create the directories or run the script again to generate dummy data.")
            exit()

        self.current_index = 0
        # Create the figure object, but not the axes yet. The axes will be created dynamically.
        self.fig = plt.figure(figsize=(18, 6))
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
        prediction_path = os.path.join(self.prediction_dir, png_filename)

        # Clear the entire figure to remove old axes before drawing new ones
        self.fig.clear()

        try:
            base_image = Image.open(img_path).convert("RGB")
            scribble_image = Image.open(scribble_path).convert("RGB")


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
            
            # --- Dynamically build a list of images to display ---
            plots_to_show = []
            plots_to_show.append({'image': overlaid_image, 'title': "Image + Scribble Overlay"})

            if os.path.exists(truth_path):
                truth_image = Image.open(truth_path)
                plots_to_show.append({'image': truth_image, 'title': "Ground Truth"})

            if os.path.exists(prediction_path):
                prediction_image = Image.open(prediction_path)
                plots_to_show.append({'image': prediction_image, 'title': "Prediction"})

            # --- Create a grid with the exact number of subplots needed ---
            num_plots = len(plots_to_show)
            axes = self.fig.subplots(1, num_plots)

            # If there's only one plot, subplots() returns a single Axes object, not an array.
            # We wrap it in a list to make the code consistent for iterating.
            if num_plots == 1:
                axes = [axes]
            
            # Display each image on its corresponding axis
            for i, p in enumerate(plots_to_show):
                axes[i].imshow(p['image'])
                axes[i].set_title(p['title'])
                axes[i].axis('off')

            self.fig.suptitle(f"File: {base_name} (.jpg/.png) ({self.current_index + 1}/{len(self.files)})", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.fig.canvas.draw()

        except FileNotFoundError as e:
            print(f"\nError: Could not find a required file.")
            print(f"Attempted to load:\n - Image: {img_path}\n - Scribble: {scribble_path}")
            print(f"Original error: {e}")
            # Create a single axis to display the error message
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"File not found:\n{os.path.basename(e.filename)}", ha='center', va='center', color='red', wrap=True)
            ax.axis('off')
            self.fig.canvas.draw()

    def run(self):
        plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize image scribbles, ground truth, and predictions.")
    parser.add_argument(
        '--root_dir', 
        type=str, 
        default="data", 
        help="The root directory containing the 'images', 'scribbles', 'ground_truth', and 'predictions' subfolders."
    )
    args = parser.parse_args()

    # --- Define paths based on the command-line argument ---
    ROOT_DIR = args.root_dir
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    SCRIBBLE_DIR = os.path.join(ROOT_DIR, "scribbles")
    TRUTH_DIR = os.path.join(ROOT_DIR, "ground_truth")
    PREDICTION_DIR = os.path.join(ROOT_DIR, "predictions")

    print(f"Using data from root directory: {os.path.abspath(ROOT_DIR)}")
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    
    visualizer = ImageVisualizer(IMAGE_DIR, SCRIBBLE_DIR, TRUTH_DIR, PREDICTION_DIR)
    visualizer.run()