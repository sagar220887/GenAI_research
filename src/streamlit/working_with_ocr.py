import pytesseract
from PIL import Image
import os
import pathlib

# Specify the path to the Tesseract executable if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

def extract_text_from_image(image_path):
    # Open the image file
    try:
        with Image.open(image_path) as img:
            # Use pytesseract to do OCR on the image
            text = pytesseract.image_to_string(img)
            return text
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

if __name__ == "__main__":
    # Specify the path to your OCR file (image)
    current_path = os.getcwd()
    image_file = os.path.join(os.path.dirname(current_path), 'data', 'ocr', 'ocr_image.png')
    print('image path - ' + image_file)
    extracted_text = extract_text_from_image(image_file)
    
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
