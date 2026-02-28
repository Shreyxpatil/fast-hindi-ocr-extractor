# Fast Hindi OCR Extractor API

A fast, FastAPI-based dual-layer text extraction server optimized for Hindi (Devanagari) and English across multiple document formats. 

## Features

- **Dual-Layer Extraction Strategy**:
  1. **Digital Text Extraction (Instant)**: Instantly extracts text from digitally born PDFs, DOCX, and TXT files.
  2. **Tesseract OCR Fallback (Fast)**: Uses OCR for scanned PDFs, images, and documents with corrupted/garbage text layers (e.g., KrutiDev-encoded text).
- **Format Support**: Supports `PDF`, `DOC` (Legacy Word), `DOCX` (Word), `TXT`, `PNG`, `JPG`, `JPEG`.
- **Image and Table Extraction**: Automatically extracts images and tabular data from documents into `Base64` format.
- **Smart Garbage Detection**: Heuristics detect whether a text layer is corrupted, forcing OCR if needed.

## Setup Instructions

### 1. Install System Dependencies (Ubuntu/Debian)
The OCR relies on `tesseract-ocr`. Ensure it is installed with Hindi language data.
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-hin libreoffice
```
*(Note: `libreoffice` is primarily used to process legacy `.doc` files or fix garbled `.docx` files by converting them to PDF)*

### 2. Install Python Dependencies
It is recommended to use a virtual environment.
```bash
python3 -m venv ocrenv
source ocrenv/bin/activate
pip install -r requirements.txt
```

### 3. Run the API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
This will start the FastAPI server on `http://localhost:8000`.

## API Usage

The extractor provides a single `POST` endpoint `/extract` expecting multipart form data.

### Request

- `file`: The document to extract (PDF, Image, Word document, etc.)
- `document_id`: A unique identifier string for this document.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_document.pdf" \
  -F "document_id=doc_12345"
```

### Response Output

The API saves and responds with a JSON structure containing:
- `all_text`: A list of strings, with each string representing the extracted layout text per page.
- `all_metadata`: Document metadata mapping to each page.
- `all_tables`: Extracted tables structured with text representation and `Base64` cropped image path.
- `all_images`: Extracted embedded images in `Base64` cropped image path format.

You can view a sample JSON output structure inside `outputs/string.json`.
