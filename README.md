# STK Produktion GmbH Knowledge Extraction

This project demonstrates automated extraction of business blocks (entities), attributes, and relationships from text, images, and audio, using Gemini (Google Generative AI), Whisper (OpenAI), and Tesseract OCR. The goal is to produce a structured "knowledge graph" for business and industrial contexts.

## Features

- Extracts structured information from text files, images (via OCR), and audio (via transcription).
- Uses Google Gemini's function calling to identify entities, attributes, and relationships in unstructured data.
- Outputs the structured results as a readable knowledge graph in the console.
- Example business scenario based on STK Produktion GmbH.

## Requirements

- Python 3.8+
- Packages: `google-generativeai`, `openai-whisper`, `pydantic`, `python-dotenv`, `numpy`, `scipy`, `pytesseract`, `Pillow`
- Tesseract OCR (must be installed and in PATH)
- Gemini API key (.env file)

## Setup

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Create a `.env` file in the project directory with your Gemini API key:
    ```
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

3. (Optional) Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract

## Usage

Run the script: Task_3.py

This will:
- Generate and extract from a sample business report (text), a PNG image (production capacity), and a dummy audio file.
- Print extracted entities and relationships in the console.

## Output

The script displays the extracted entities (blocks), their attributes, and the relationships between them in a clear text format.

## Notes

- Tesseract OCR is required for image text extraction (make sure it's in your PATH).
- The audio demo uses a silent file as placeholder if you do not supply real audio.
- Result parsing expects the Gemini API to return valid structured data.

## License

MIT License.

## Credits

- Google Generative AI Python SDK
- OpenAI Whisper
- Tesseract OCR
