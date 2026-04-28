"""
One-shot script to generate the sample.pdf fixture used by test_ingestion.py.

Run once with:
    pip install reportlab
    python tests/fixtures/_generate_sample.py

The resulting sample.pdf is committed to the repo. reportlab is NOT a
project dependency — it's only used to (re)generate this fixture if needed.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


OUTPUT = Path(__file__).parent / "sample.pdf"


def main() -> None:
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4)

    # Page 1 — Italian content
    pdf.setFont("Helvetica", 14)
    pdf.drawString(72, 800, "Documento di test per il RAG chatbot")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, 760, "Il gatto domestico dorme tranquillamente sul divano del salotto.")
    pdf.drawString(72, 740, "La pasta al pomodoro e' una ricetta tradizionale italiana.")
    pdf.drawString(72, 720, "Roma e' la capitale d'Italia ed e' famosa per il Colosseo.")
    pdf.showPage()

    # Page 2 — English content (test multilingual)
    pdf.setFont("Helvetica", 14)
    pdf.drawString(72, 800, "Test document for the RAG chatbot")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(72, 760, "The domestic cat sleeps peacefully on the living room couch.")
    pdf.drawString(72, 740, "Tomato pasta is a traditional Italian recipe.")
    pdf.drawString(72, 720, "Rome is the capital of Italy and is famous for the Colosseum.")
    pdf.showPage()

    pdf.save()
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()