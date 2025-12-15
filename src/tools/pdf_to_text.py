
import argparse
import os
from PyPDF2 import PdfReader

def convert_pdf_to_text(pdf_path):
    """
    Extracts text from a PDF file, removes line breaks,
    and saves it to a text file with the same name.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # Replace line breaks with spaces
        text = text.replace('\n', ' ').replace('\r', '')

        txt_path = os.path.splitext(pdf_path)[0] + ".txt"
        
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)
            
        print(f"Successfully converted {pdf_path} to {txt_path}")

    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")


def main():
    """
    Main function to handle command-line arguments and process files.
    """
    parser = argparse.ArgumentParser(
        description="Convert PDF files to text without line breaks."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a PDF file or a directory containing PDF files."
    )
    args = parser.parse_args()

    input_path = args.path

    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
        return

    if os.path.isfile(input_path):
        if input_path.lower().endswith(".pdf"):
            convert_pdf_to_text(input_path)
        else:
            print("Error: Input file is not a PDF.")
    elif os.path.isdir(input_path):
        for item in os.listdir(input_path):
            if item.lower().endswith(".pdf"):
                file_path = os.path.join(input_path, item)
                convert_pdf_to_text(file_path)

if __name__ == "__main__":
    main()
