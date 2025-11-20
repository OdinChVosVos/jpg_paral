import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import jpeg_py
import time

class JPEGEncoderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Parallel Encoder")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")
        
        self.quality = 85
        self.threads = 4
        self.encoder = jpeg_py.JPEGEncoder(self.quality, self.threads)
        self.current_image = None
        
        self.setup_ui()
        self.generate_test_image()
        
    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#3c3c3c", pady=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(control_frame, text="Качество:", 
                bg="#3c3c3c", fg="white", font=("Arial", 12)).grid(row=0, column=0, padx=10)
        
        self.quality_var = tk.IntVar(value=self.quality)
        quality_slider = ttk.Scale(control_frame, from_=10, to=100, 
                                   variable=self.quality_var, orient=tk.HORIZONTAL,
                                   length=200, command=self.on_quality_change)
        quality_slider.grid(row=0, column=1, padx=5)
        
        self.quality_label = tk.Label(control_frame, text=str(self.quality),
                                     bg="#3c3c3c", fg="#4CAF50", font=("Arial", 12, "bold"))
        self.quality_label.grid(row=0, column=2, padx=5)
        
        tk.Label(control_frame, text="Потоки:", 
                bg="#3c3c3c", fg="white", font=("Arial", 12)).grid(row=0, column=3, padx=10)
        
        self.threads_var = tk.IntVar(value=self.threads)
        threads_slider = ttk.Scale(control_frame, from_=1, to=16, 
                                  variable=self.threads_var, orient=tk.HORIZONTAL,
                                  length=150, command=self.on_threads_change)
        threads_slider.grid(row=0, column=4, padx=5)
        
        self.threads_label = tk.Label(control_frame, text=str(self.threads),
                                     bg="#3c3c3c", fg="#2196F3", font=("Arial", 12, "bold"))
        self.threads_label.grid(row=0, column=5, padx=5)
        
        btn_frame = tk.Frame(self.root, bg="#2b2b2b")
        btn_frame.pack(pady=10)
        
        patterns = [("Градиент", 0), ("Шахматы", 1), ("Волны", 2)]
        for text, pattern in patterns:
            btn = tk.Button(btn_frame, text=text, 
                          command=lambda p=pattern: self.generate_test_image(p),
                          bg="#4CAF50", fg="white", font=("Arial", 11),
                          padx=20, pady=5, relief=tk.RAISED)
            btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Сжать", command=self.compress_image,
                 bg="#FF9800", fg="white", font=("Arial", 11, "bold"),
                 padx=30, pady=5, relief=tk.RAISED).pack(side=tk.LEFT, padx=5)
        
        self.stats_frame = tk.Frame(self.root, bg="#3c3c3c", pady=10)
        self.stats_frame.pack(fill=tk.X, padx=10)
        
        self.time_label = tk.Label(self.stats_frame, text="Время: -",
                                   bg="#3c3c3c", fg="#FFC107", font=("Arial", 11))
        self.time_label.pack()
        
        self.size_label = tk.Label(self.stats_frame, text="Размер: -",
                                   bg="#3c3c3c", fg="#FFC107", font=("Arial", 11))
        self.size_label.pack()
        
        self.image_label = tk.Label(self.root, bg="#1e1e1e")
        self.image_label.pack(expand=True, pady=20)
        
    def on_quality_change(self, value):
        self.quality = int(float(value))
        self.quality_label.config(text=str(self.quality))
        self.encoder.set_quality(self.quality)
        
    def on_threads_change(self, value):
        self.threads = int(float(value))
        self.threads_label.config(text=str(self.threads))
        self.encoder.set_threads(self.threads)
        
    def generate_test_image(self, pattern=0):
        size = 512
        img_array = np.zeros((size, size, 3), dtype=np.uint8)
        
        if pattern == 0:
            for y in range(size):
                for x in range(size):
                    img_array[y, x] = [
                        int(255 * x / size),
                        int(255 * y / size),
                        128
                    ]
        elif pattern == 1:
            for y in range(size):
                for x in range(size):
                    checker = ((x // 32) + (y // 32)) % 2
                    val = 255 if checker else 64
                    img_array[y, x] = [val, val, val]
        elif pattern == 2:
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt((x - size/2)**2 + (y - size/2)**2)
                    val = int(np.sin(dist * 0.1) * 127 + 128)
                    img_array[y, x] = [val, 255 - val, val // 2]
        
        self.current_image = img_array
        self.display_image(img_array)
        
    def compress_image(self):
        if self.current_image is None:
            return
            
        img = jpeg_py.Image()
        img.width = self.current_image.shape[1]
        img.height = self.current_image.shape[0]
        img.channels = 3
        img.data = self.current_image.flatten().tolist()
        
        start = time.time()
        encoded = self.encoder.encode(img)
        duration = (time.time() - start) * 1000
        
        self.time_label.config(text=f"Время сжатия: {duration:.2f} мс")
        self.size_label.config(text=f"Размер: {encoded.size / 1024:.2f} КБ")
        
    def display_image(self, img_array):
        img = Image.fromarray(img_array)
        img.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGEncoderGUI(root)
    root.mainloop()
