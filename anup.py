
import os
import io
import uuid
import base64
import traceback
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QUADRANT_ENDPOINT = os.getenv("QUADRANT_ENDPOINT")
QUADRANT_API_KEY = os.getenv("QUADRANT_API_KEY")


llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0
)
embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


text_prompt = PromptTemplate(
    input_variables=["block"],
    template="Summarize the following text block concisely:\n\n{block}"
)
table_prompt = PromptTemplate(
    input_variables=["block"],
    template="Summarize the following table in plain English:\n\n{block}"
)
image_prompt = PromptTemplate(
    input_variables=["block"],
    template="Summarize the following OCR text from an image:\n\n{block}"
)
text_chain = text_prompt | llm
table_chain = table_prompt | llm
image_chain = image_prompt | llm


splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, length_function=len)


def pil_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def advanced_extract_images(pdf_bytes: bytes, pdf_filename: str) -> List[Tuple[Image.Image, dict]]:
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            meta = {"source": pdf_filename, "page": page_idx+1, "type": "high_res"}
            images.append((img, meta))
        doc.close()
    except Exception as e:
        print(f"[ERROR] advanced_extract_images: {e}")
    return images

# Ingest PDF
def ingest_pdf_bytes(pdf_bytes: bytes, pdf_filename: str) -> Tuple[int, List[str]]:
    summaries: List[Document] = []
    errors: List[str] = []
    temp_path = f"/tmp/{uuid.uuid4()}_{pdf_filename}"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)
    try:
        # Full-page text
        for idx, page in enumerate(PyPDFLoader(temp_path).load(), 1):
            text = page.page_content.strip()
            if text:
                try:
                    resp = text_chain.invoke({"block": text})
                    summaries.append(Document(
                        page_content=resp.content,
                        metadata={"source": pdf_filename, "page": idx, "type": "page", "raw": text}
                    ))
                except Exception as e:
                    errors.append(f"Page {idx} summary failed: {e}")
        # Elements
        elements = UnstructuredPDFLoader(temp_path, mode="elements").load()
        text_chunks, tables = [], []
        for el in elements:
            pg = el.metadata.get("page_number")
            md = {"source": pdf_filename, "page": pg}
            if el.metadata.get("element_type", "").startswith("Narrative"):
                text_chunks.append((el.page_content, md))
            elif el.metadata.get("element_type") == "Table":
                tables.append((el.page_content, md))
        # Text chunks
        for content, md in text_chunks:
            for chunk in splitter.split_text(content):
                try:
                    resp = text_chain.invoke({"block": chunk})
                    summaries.append(Document(
                        page_content=resp.content,
                        metadata={**md, "type": "chunk", "raw": chunk}
                    ))
                except Exception as e:
                    errors.append(f"Chunk summary failed: {e}")
        # Tables
        for tbl, md in tables:
            block = f"<table>\n{tbl}\n</table>"
            try:
                resp = table_chain.invoke({"block": block})
                summaries.append(Document(
                    page_content=resp.content,
                    metadata={**md, "type": "table", "raw": tbl}
                ))
            except Exception as e:
                errors.append(f"Table summary failed: {e}")
        # Images OCR
        images = advanced_extract_images(pdf_bytes, pdf_filename)
        for pil_img, md in images:
            ocr = pytesseract.image_to_string(pil_img).strip()
            if not ocr:
                continue
            try:
                img_b64 = pil_to_base64(pil_img)
                resp = image_chain.invoke({"block": ocr})
                summaries.append(Document(
                    page_content=resp.content,
                    metadata={**md, "type": "image", "raw": ocr, "img_b64": img_b64}
                ))
            except Exception as e:
                errors.append(f"Image summary failed: {e}")
    except Exception as e:
        errors.append(str(e))
    finally:
        os.remove(temp_path)
    # Qdrant
    client = QdrantClient(url=QUADRANT_ENDPOINT, api_key=QUADRANT_API_KEY)
    if not client.collection_exists("pdf_multimodal_summaries"):
        client.create_collection(
            collection_name="pdf_multimodal_summaries",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    store = QdrantVectorStore(client=client, collection_name="pdf_multimodal_summaries", embedding=embedder)
    if summaries:
        store.add_documents(summaries)
    return len(summaries), errors
