import os
import traceback
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from anup import ingest_pdf_bytes
from retrieval import PDFRAGRetriever
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(
    title="Multi-PDF Q&A API",
    description="Send PDFs to /upload, then chat via /ask with citations and images.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever: Optional[PDFRAGRetriever] = None

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0
)

COLLECTION = "pdf_multimodal_summaries"

class Query(BaseModel):
    text: str

class UploadResponse(BaseModel):
    status: str
    chunks_indexed: int
    errors: List[str]

class ImageResponse(BaseModel):
    img_b64: str
    source: str
    page: int

class AnswerResponse(BaseModel):
    answer: str
    citations: List[str]
    images: List[ImageResponse]

@app.get("/")
def root():
    return {"message": "API up. See /docs"}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    files: List[UploadFile] = File(...),
):
    global retriever
    client = QdrantClient(
        url=os.getenv("QUADRANT_ENDPOINT"),
        api_key=os.getenv("QUADRANT_API_KEY")
    )
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    total_chunks = 0
    errors: List[str] = []
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            errors.append(f"{upload.filename}: not a PDF")
            continue
        try:
            pdf_bytes = await upload.read()
            n, errs = ingest_pdf_bytes(pdf_bytes, upload.filename)
            total_chunks += n
            errors.extend(errs)
        except Exception as e:
            tb = traceback.format_exc()
            errors.append(f"{upload.filename} ingestion error: {e}")
            print(f"[ERROR] ingesting {upload.filename}\n{tb}")
    # Initialize retriever after ingestion
    retriever = PDFRAGRetriever(
        qdrant_endpoint=os.getenv("QUADRANT_ENDPOINT"),
        qdrant_api_key=os.getenv("QUADRANT_API_KEY"),
        collection_name=COLLECTION
    )
    return UploadResponse(status="success", chunks_indexed=total_chunks, errors=errors)

@app.post("/ask", response_model=AnswerResponse)
def ask(query: Query):
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Empty question.")
    if retriever is None:
        raise HTTPException(status_code=400, detail="No PDFs indexed. POST /upload first.")

    try:
        hits = retriever.hybrid_retrieve(query.text, top_k=3)
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] Retrieval error\n", tb)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    if not hits:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    # Build prompt and collect citations/images
    blocks: List[str] = []
    citations: List[str] = []
    images_resp: List[ImageResponse] = []
    for idx, h in enumerate(hits, start=1):
        src = h.get("source") or h.get("metadata", {}).get("source", "unknown")
        pg = h.get("page") or h.get("metadata", {}).get("page", -1)
        citations.append(f"{src} (page {pg})")
        summary = h.get("summary", h.get("metadata", {}).get("raw", ""))
        raw = h.get("raw", "")
        blocks.append(f"Context {idx}:\nSummary: {summary}\nRaw: {raw}")
        # Include images if any
        img_b64 = h.get("img_b64") or h.get("metadata", {}).get("img_b64")
        if img_b64:
            images_resp.append(ImageResponse(img_b64=img_b64, source=src, page=pg))

    prompt = (
        "You are a helpful assistant. Use the following contexts to answer the user's question.\n\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion: {query.text}\nAnswer:"
    )

    try:
        llm_resp = llm.invoke(prompt)
        answer_text = llm_resp.content.strip()
    except Exception as e:
        tb = traceback.format_exc()
        print("[ERROR] LLM error\n", tb)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return AnswerResponse(answer=answer_text, citations=citations, images=images_resp)
