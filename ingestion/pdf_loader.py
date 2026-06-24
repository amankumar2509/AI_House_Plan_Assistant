from pypdf import PdfReader


def load_pdf(pdf_path):
    """
    Load PDF and return extracted text.
    """

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text