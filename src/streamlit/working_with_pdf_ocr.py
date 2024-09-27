import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    extracted_text = ""

    # Iterate through each page
    for page in doc:
        extracted_text += page.get_text() + "\n"

    return extracted_text

if __name__ == "__main__":
    # Specify the path to your PDF file
    current_path = os.getcwd()
    pdf_file = os.path.join(os.path.dirname(current_path), 'data', 'ocr', 'pdf_scanned_ocr.pdf')
    extracted_text = extract_text_from_pdf(pdf_file)
    
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
