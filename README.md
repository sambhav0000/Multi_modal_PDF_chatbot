# Multi-Modal PDF Chatbot

The **Multi-Modal PDF Chatbot** allows users to upload PDF files, automatically extract and summarize text, tables, and images, and chat with documents through a conversational interface with cited sources and OCR image extraction.

---

## Table of Contents

* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Key Components](#key-components)

  * [Ingestion](#ingestion)
  * [Retrieval](#retrieval)
  * [Web API](#web-api)
  * [Frontend](#frontend)
* [Environment Variables](#environment-variables)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Features

* **PDF Upload and Indexing**: Upload PDFs, extract full-page text, split content, process tables and images (with OCR), and generate summaries.
* **Conversational Q\&A**: Ask questions to get answers with supporting citations from the document content.
* **Hybrid Retrieval**: Semantic search via vector embeddings with keyword fallback.
* **Interactive Frontend**: A [Streamlit](https://streamlit.io/) UI with chat history and image display.
* **Scalable**: FastAPI + Qdrant for efficient API and vector storage.

---

## Project Structure

```
Multi_modal_PDF_chatbot/
├── .env                  # Environment variables
├── .gitignore            # Ignored files
├── ingestion.py          # PDF ingestion and summarization
├── retrieval_api.py      # FastAPI app
├── retrieval.py          # PDFRAGRetriever implementation
├── retriever_setup.py    # Qdrant retriever setup
├── streamlit_app.py      # Streamlit frontend
├── raw_documents.db      # Shelve store for raw text
└── pdfs/                 # Sample PDFs
```

---

## Installation

1. **Clone** the repo:

   ```bash
   git clone https://github.com/your-repo/Multi_modal_PDF_chatbot.git
   cd Multi_modal_PDF_chatbot
   ```

2. **Create & activate venv**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure `.env`**:

   ```ini
   OPENAI_API_KEY=your_openai_api_key
   QUADRANT_ENDPOINT=http://localhost:6333
   QUADRANT_API_KEY=
   QDRANT_INDEX_NAME=pdf_multimodal_summaries
   API_URL=http://localhost:8000
   ```

---

## Usage

### Start Backend

```bash
uvicorn retrieval_api:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Open [http://localhost:8501](http://localhost:8501) to interact.

---

## Key Components

### Ingestion (`ingestion.py`)

* Extracts text/tables/images, summarizes with OpenAI, indexes into Qdrant.

### Retrieval (`retrieval.py` & `retriever_setup.py`)

* `PDFRAGRetriever` for semantic + keyword search.

### Web API (`retrieval_api.py`)

* FastAPI endpoints `/search` (POST) and `/health` (GET).
* Integrates retriever + OpenAI chat model.

### Frontend (`streamlit_app.py`)

* Streamlit UI for PDF upload and chat.

---

## Environment Variables

| Variable            | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `OPENAI_API_KEY`    | OpenAI API key for embeddings & chat.                                  |
| `QUADRANT_ENDPOINT` | Qdrant URL (local or cloud).                                           |
| `QUADRANT_API_KEY`  | Qdrant API key (if required).                                          |
| `QDRANT_INDEX_NAME` | Collection name in Qdrant.                                             |
| `API_URL`           | Backend URL (default: [http://localhost:8000](http://localhost:8000)). |

---

## Troubleshooting

* **Port in use**: Change ports or kill existing processes.
* **Env vars missing**: Confirm `.env` is loaded and variables are correct.
* **Qdrant not running**: Start via Docker: `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant`.

---

## Contributing

1. Fork.
2. Create branch.
3. PR with details.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

* FastAPI, Streamlit, Qdrant, OpenAI, PyMuPDF, UnstructuredPDFLoader, pytesseract.
