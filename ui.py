import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from ocr import process_image
from pdf2image import convert_from_path

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST OCR GUI")
        
        screen_w = int(root.winfo_screenwidth() * 0.5)
        screen_h = int(root.winfo_screenheight() * 0.5)
        self.root.geometry(f"{screen_w}x{screen_h}")
        
        self.current_images = []
        self.current_page = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        upload_btn = tk.Button(btn_frame, text="Upload Image / PDF", command=self.upload_file)
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.prev_btn = tk.Button(btn_frame, text="< Prev", command=self.prev_page, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.page_label = tk.Label(btn_frame, text="Page: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(btn_frame, text="Next >", command=self.next_page, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        self.canvas = tk.Canvas(self.root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def upload_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images and PDFs", "*.png *.jpg *.jpeg *.pdf")]
        )
        if not file_path:
            return
            
        self.current_images = []
        self.current_page = 0
        
        if file_path.lower().endswith('.pdf'):
            try:
                pages = convert_from_path(file_path)
                for i, page in enumerate(pages):
                    temp_path = f"temp_page_{i}.jpg"
                    page.save(temp_path, 'JPEG')
                    
                    processed_img = process_image(temp_path)
                    if processed_img is not None:
                        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                        self.current_images.append(Image.fromarray(rgb_img))
                        
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process PDF: {e}\nEnsure poppler is installed.")
                return
        else:
            processed_img = process_image(file_path)
            if processed_img is not None:
                rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                self.current_images.append(Image.fromarray(rgb_img))
            else:
                messagebox.showerror("Error", "Model not found. Run train.py first.")
                return
                
        self.update_view()
        
    def update_view(self):
        if not self.current_images:
            return
            
        total_pages = len(self.current_images)
        self.page_label.config(text=f"Page: {self.current_page + 1}/{total_pages}")
        
        self.prev_btn.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_page < total_pages - 1 else tk.DISABLED)
        
        img = self.current_images[self.current_page]
        
        c_width = self.canvas.winfo_width()
        c_height = self.canvas.winfo_height()
        
        if c_width <= 1 or c_height <= 1:
            c_width, c_height = 800, 600
            
        img.thumbnail((c_width, c_height), Image.Resampling.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(img) # Keep reference
        self.canvas.delete("all")
        self.canvas.create_image(c_width//2, c_height//2, image=self.tk_img, anchor=tk.CENTER)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_view()

    def next_page(self):
        if self.current_page < len(self.current_images) - 1:
            self.current_page += 1
            self.update_view()

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.update()
    root.mainloop()
