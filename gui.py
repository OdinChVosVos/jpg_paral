import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import jpeg_py
import time
import os

class JPEGEncoderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JPEG Parallel Encoder")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2b2b2b")
        
        self.max_threads = os.cpu_count()
        self.quality = 85
        self.threads = 4
        self.encoder = jpeg_py.JPEGEncoder(self.quality, self.threads)
        self.current_image = None
        self.compressed_image = None
        self.loaded_filename = None
        
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
        
        tk.Button(control_frame, text="Авто", 
                 command=self.set_auto_threads,
                 bg="#03A9F4", fg="white", font=("Arial", 10),
                 padx=10, pady=3).grid(row=0, column=6, padx=5)
        
        btn_frame = tk.Frame(self.root, bg="#2b2b2b")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Загрузить изображение", 
                 command=self.load_image,
                 bg="#9C27B0", fg="white", font=("Arial", 11, "bold"),
                 padx=20, pady=5, relief=tk.RAISED).pack(side=tk.LEFT, padx=5)
        
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
        
        self.filename_label = tk.Label(self.stats_frame, text="Файл: Тестовое изображение",
                                      bg="#3c3c3c", fg="#E0E0E0", font=("Arial", 10))
        self.filename_label.pack()
        
        self.time_label = tk.Label(self.stats_frame, text="Время: -",
                                   bg="#3c3c3c", fg="#FFC107", font=("Arial", 11))
        self.time_label.pack()
        
        self.size_label = tk.Label(self.stats_frame, text="Размер: -",
                                   bg="#3c3c3c", fg="#FFC107", font=("Arial", 11))
        self.size_label.pack()
        
        self.compression_label = tk.Label(self.stats_frame, text="Нулевых коэф.: -",
                                         bg="#3c3c3c", fg="#FFC107", font=("Arial", 11))
        self.compression_label.pack()
        
        images_frame = tk.Frame(self.root, bg="#2b2b2b")
        images_frame.pack(expand=True, pady=10)
        
        left_frame = tk.Frame(images_frame, bg="#1e1e1e")
        left_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(left_frame, text="Оригинал", bg="#1e1e1e", fg="white",
                font=("Arial", 14, "bold")).pack(pady=5)
        self.original_label = tk.Label(left_frame, bg="#1e1e1e")
        self.original_label.pack()
        
        right_frame = tk.Frame(images_frame, bg="#1e1e1e")
        right_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Label(right_frame, text="После сжатия", bg="#1e1e1e", fg="white",
                font=("Arial", 14, "bold")).pack(pady=5)
        self.compressed_label = tk.Label(right_frame, bg="#1e1e1e")
        self.compressed_label.pack()
        
    def set_auto_threads(self):
        self.threads = self.max_threads
        self.threads_var.set(self.threads)
        self.threads_label.config(text=str(self.threads))
        self.encoder.set_threads(self.threads)
        print(f"Установлено потоков: {self.threads} (все доступные)")
        
    def on_quality_change(self, value):
        self.quality = int(float(value))
        self.quality_label.config(text=str(self.quality))
        self.encoder.set_quality(self.quality)
        
    def on_threads_change(self, value):
        self.threads = int(float(value))
        self.threads_label.config(text=str(self.threads))
        self.encoder.set_threads(self.threads)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            initialdir=os.path.expanduser("~"),
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Все файлы", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            img = Image.open(file_path)
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            max_size = 1024
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                messagebox.showinfo("Изменение размера", 
                                   f"Изображение уменьшено до {img.width}×{img.height}\n"
                                   f"(максимум {max_size}×{max_size} для производительности)")
            
            img_array = np.array(img, dtype=np.uint8)
            
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            self.current_image = img_array
            self.loaded_filename = os.path.basename(file_path)
            self.filename_label.config(text=f"Файл: {self.loaded_filename} ({img.width}×{img.height})")
            
            self.display_original(img_array)
            self.compressed_image = None
            self.compressed_label.config(image='')
            
            print(f"Загружено: {file_path}")
            print(f"Размер: {img_array.shape}")
            
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить изображение:\n{str(e)}")
            print(f"Ошибка: {e}")
        
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
            pattern_name = "Градиент"
        elif pattern == 1:
            for y in range(size):
                for x in range(size):
                    checker = ((x // 32) + (y // 32)) % 2
                    val = 255 if checker else 64
                    img_array[y, x] = [val, val, val]
            pattern_name = "Шахматы"
        elif pattern == 2:
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt((x - size/2)**2 + (y - size/2)**2)
                    val = int(np.sin(dist * 0.1) * 127 + 128)
                    img_array[y, x] = [val, 255 - val, val // 2]
            pattern_name = "Волны"
        
        self.current_image = img_array
        self.loaded_filename = None
        self.filename_label.config(text=f"Файл: Тестовое ({pattern_name})")
        self.display_original(img_array)
        self.compressed_image = None
        self.compressed_label.config(image='')
        
    def compress_image(self):
        if self.current_image is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите или сгенерируйте изображение")
            return
            
        img = jpeg_py.Image()
        img.width = self.current_image.shape[1]
        img.height = self.current_image.shape[0]
        img.channels = 3
        img.data = self.current_image.flatten().tolist()
        
        start = time.time()
        encoded = self.encoder.encode(img)
        decoded = self.encoder.decode(encoded)
        duration = (time.time() - start) * 1000
        
        self.compressed_image = np.array(decoded.data).reshape(
            decoded.height, decoded.width, decoded.channels
        ).astype(np.uint8)
        
        zero_percent = 100.0 * encoded.zero_coefficients / encoded.total_coefficients
        compression_ratio = 100.0 * (1.0 - encoded.size / encoded.uncompressed_size)
        
        self.time_label.config(text=f"Время сжатия: {duration:.2f} мс")
        self.size_label.config(
            text=f"Размер: {encoded.uncompressed_size/1024:.1f} КБ → {encoded.size/1024:.1f} КБ ({compression_ratio:.1f}% сжатие)"
        )
        self.compression_label.config(
            text=f"Нулевых коэф.: {encoded.zero_coefficients}/{encoded.total_coefficients} ({zero_percent:.1f}%)"
        )
        
        self.display_compressed(self.compressed_image)

        
    def display_original(self, img_array):
        img = Image.fromarray(img_array)
        img.thumbnail((450, 450), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.original_label.config(image=photo)
        self.original_label.image = photo
        
    def display_compressed(self, img_array):
        img = Image.fromarray(img_array)
        img.thumbnail((450, 450), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.compressed_label.config(image=photo)
        self.compressed_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = JPEGEncoderGUI(root)
    root.mainloop()
