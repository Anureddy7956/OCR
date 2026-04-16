import os
import sys
from pdf2image import convert_from_path
import cv2
import numpy as np

from ocr import process_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_ocr.py <pdf_path>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
        
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = f"{pdf_name}_ocr_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting PDF: {pdf_path}")
    try:
        pages = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Failed to convert PDF. Do you have poppler installed? Error: {e}")
        sys.exit(1)
        
    print(f"Processing {len(pages)} pages...")
    for i, page in enumerate(pages):
        page_img_path = os.path.join(output_dir, f"page_{i+1}_raw.jpg")
        page.save(page_img_path, 'JPEG')
        
        output_save_path = os.path.join(output_dir, f"page_{i+1}_ocr.jpg")
        print(f"Running OCR on page {i+1}...")
        process_image(page_img_path, save_path=output_save_path)
        
    print(f"Done processing {pdf_path}. Check {output_dir}/")

if __name__ == '__main__':
    main()
