# retriever_setup.py

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# 1) Connect to your Qdrant deployment
client = QdrantClient(
    url=os.environ["QDRANT_ENDPOINT"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# 2) Instantiate the store for retrieval only
vector_store = QdrantVectorStore(
    client=client,
    collection_name=os.environ["QDRANT_INDEX_NAME"],
)

# 3) Build a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
