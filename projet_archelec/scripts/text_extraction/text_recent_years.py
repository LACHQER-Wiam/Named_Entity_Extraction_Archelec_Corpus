from pathlib import Path
import pymupdf

def ocr_files(file_path: Path, output_file: Path):
    """
    Extract text from pdf files and save them as .txt files.
    """
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print(f"Extracted text from {file_path} and saved to {output_file}")
    # Save the extracted text to a .txt file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == "__main__":

    folders = ['RG15', 'LG17', 'ER19', 'LG20', 'MN20']

    for folder in folders:
        print(f"Processing folder: {folder}")
        for num in range(1, 11):
            folder_path = Path(f"./data/raw/recent_pdf/{folder}")
            file_path = folder_path / f"{folder}_{num}.pdf"
            output_path = Path(f"./data/raw/arkindex_archelec/text_files/recent_years/{folder}_{num}.txt")
            ocr_files(file_path, output_path)