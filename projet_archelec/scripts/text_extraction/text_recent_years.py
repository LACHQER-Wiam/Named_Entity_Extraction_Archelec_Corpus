from pathlib import Path
import pymupdf
import pytesseract
from PIL import Image
import io
import json
import re
import ftfy

MIN_CHARS = 50


def extract_text(file_path: Path) -> str:
    doc = pymupdf.open(file_path)
    text = "".join(page.get_text() for page in doc)

    if len(text.strip()) >= MIN_CHARS:
        return text

    # OCR fallback
    print(f"  → OCR fallback for {file_path.name}")
    text = ""
    for page in doc:
        mat = pymupdf.Matrix(300 / 72, 300 / 72)
        img = Image.open(io.BytesIO(page.get_pixmap(matrix=mat).tobytes("png")))
        text += pytesseract.image_to_string(img, lang="fra")
    return text

def clean_text(text: str) -> str:
    # Corriger l'encodage avec ftfy
    text_clean = ftfy.fix_text(text)

    # Nettoyer
    text_clean = re.sub(r"\s+", " ", text_clean).strip()
    text_clean = re.sub(r"\n{3,}", "\n\n", text_clean)

    return text_clean


if __name__ == "__main__":
    folders = ["RG15", "LG17", "ER19", "LG20", "MN20"]
    formatted_data = []

    for folder in folders:
        for num in range(1, 11):
            file_path = Path(f"./data/raw/recent_pdf/{folder}/{folder}_{num}.pdf")
            output_path = Path(f"./data/raw/arkindex_archelec/text_files/recent_years/{folder}_{num}.txt")

            text = extract_text(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")

            formatted_data.append({
                "id": f"{folder}_{num}",
                "annee": f"20{folder[2:]}",
                "texte": clean_text(text),
                "entites": []
            })

    annotation_file = Path("./data/annotated/archelec_annotated_2015_2020.json")
    annotation_file.parent.mkdir(parents=True, exist_ok=True)
    annotation_file.write_text(json.dumps(formatted_data, ensure_ascii=False, indent=2), encoding="utf-8")