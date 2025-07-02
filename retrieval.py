import os
import traceback
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


env = os.getenv

class PDFRAGRetriever:
    def __init__(
        self,
        qdrant_endpoint: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "pdf_multimodal_summaries",
    ):
        endpoint = qdrant_endpoint or env("QUADRANT_ENDPOINT")
        api_key = qdrant_api_key or env("QUADRANT_API_KEY")
        self.client = QdrantClient(url=endpoint, api_key=api_key)

        try:
            if not self.client.collection_exists(collection_name=collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
                )
                print(f"[INFO] Created collection '{collection_name}'.")
        except Exception as e:
            print(f"[ERROR] Collection setup failed: {e}")
            traceback.print_exc()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=OpenAIEmbeddings(openai_api_key=env("OPENAI_API_KEY"))
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            docs = self.vectorstore.similarity_search(query, k=top_k)
        except Exception as e:
            print(f"[ERROR] similarity_search failed: {e}")
            traceback.print_exc()
            return []

        results = []
        for doc in docs:
            md = doc.metadata
            results.append({
                "summary": doc.page_content,
                "raw": md.get("raw", ""),
                "img_b64": md.get("img_b64"),
                "source": md.get("source", ""),
                "page": md.get("page", None)
            })
        return results

    def hybrid_retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        semantic_results = self.retrieve(query, top_k)

        # Perform keyword search fallback if not enough semantic results
        if len(semantic_results) < top_k:
            keyword_results = []
            seen = {(res["source"], res["page"]) for res in semantic_results}

            # Fetch larger set of docs to check for keyword matches
            potential_docs = self.vectorstore.similarity_search("", k=50)

            for doc in potential_docs:
                md = doc.metadata
                raw_text = md.get("raw", "").lower()

                if query.lower() in raw_text:
                    identifier = (md.get("source"), md.get("page"))
                    if identifier not in seen:
                        keyword_results.append({
                            "summary": doc.page_content,
                            "raw": md.get("raw", ""),
                            "img_b64": md.get("img_b64"),
                            "source": md.get("source", ""),
                            "page": md.get("page", None)
                        })
                        seen.add(identifier)

                if len(semantic_results) + len(keyword_results) >= top_k:
                    break

            return semantic_results + keyword_results

        return semantic_results
