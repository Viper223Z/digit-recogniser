import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, UnidentifiedImageError
import os
import create_model
import predict_image  # Dodajemy import modułu predict_image

# Wygląd aplikacji
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.label = tk.Label(root, text="Digit Recognizer", font=("Helvetica", 16))
        self.label.pack(pady=20)

        self.create_model_button = tk.Button(root, text="Create Neural Network", command=self.create_model)
        self.create_model_button.pack(pady=10)

        self.load_image_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_image_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=20)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        self.image_path = None

    def create_model(self):
        self.progress["value"] = 0
        create_model.create_and_save_model(self.update_progress)

    def update_progress(self, value):
        self.progress["value"] = value
        if value == 100:
            self.root.after(0, lambda: messagebox.showinfo("Info", "Model Created and Saved Successfully"))
        else:
            self.root.update_idletasks()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            try:
                image = Image.open(self.image_path)
                image.verify()  # Verify that it is an image
                self.predict_button.config(state=tk.NORMAL)
                messagebox.showinfo("Info", "Image Loaded Successfully")
            except (UnidentifiedImageError, IOError):
                messagebox.showerror("Error", "The selected file is not a valid image.")
                self.image_path = None
                self.predict_button.config(state=tk.DISABLED)

    def predict(self):
        if not os.path.exists('my_model.keras'):
            messagebox.showerror("Error", "Model not found. Please create the neural network first.")
            return

        if self.image_path:
            try:
                predicted_number = predict_image.load_and_predict(self.image_path)
                self.result_label.config(text=f"Predicted Number: {predicted_number}")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        else:
            messagebox.showerror("Error", "Image not loaded")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
