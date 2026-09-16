import pdfplumber
import os
import sys

def extract_pdf_text(pdf_path, output_dir):
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(output_dir, base_name + ".txt")

    with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as txt_file:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
               # txt_file.write(f"--- Page {page_number} ---\n")
                txt_file.write(text + "\n\n")

 #   print(f"Saved: {txt_path}")

def process_directory(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        raise ValueError("Input path is not a directory")

    os.makedirs(output_dir, exist_ok=True)

    # Collect existing txt base names
    existing_txts = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_dir)
        if f.lower().endswith(".txt")
    }

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        base_name = os.path.splitext(filename)[0]

        # Skip if corresponding txt already exists
        if base_name in existing_txts:
            print(f"Skipping {filename} (txt already exists)")
            continue

        pdf_path = os.path.join(input_dir, filename)
        extract_pdf_text(pdf_path, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdf_to_txt.py <input_pdf_directory> <output_txt_directory>")
        sys.exit(1)

    process_directory(sys.argv[1], sys.argv[2])
