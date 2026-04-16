# MNIST OCR Project

A CNN-based handwritten digit recognition system, equipped with text extraction (OCR) capabilities for images and multi-page PDFs, and a Tkinter GUI.

## Prerequisites
- Linux / Ubuntu
- `poppler-utils` (used by `pdf2image` for PDF processing):
  ```bash
  sudo apt-get update
  sudo apt-get install poppler-utils
  ```

## Setup Environment
1. Create and activate a conda environment named `IISC_1104_env`:
   ```bash
   conda create -n env_name python=3.10 -y
   conda activate env_name
   ```
2. Install the necessary pip packages:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow
1. **Train the Model**:
   ```bash
   python train.py
   ```
   This will download MNIST, train the model, show verbose output, generate a confusion matrix, and save the model to the `model/` folder.

2. **Test the Model**:
   ```bash
   python test.py
   ```
   This evaluates the saved model on the test dataset.

3. **Predict a Single Digit**:
   ```bash
   python predict.py path/to/image.png
   ```

4. **Image OCR**:
   ```bash
   python ocr.py path/to/image.png
   ```

5. **PDF OCR**:
   ```bash
   python pdf_ocr.py path/to/file.pdf
   ```

6. **Graphical User Interface (GUI)**:
   ```bash
   python ui.py
   ```
   A graphical window allowing uploads and OCR interaction.
