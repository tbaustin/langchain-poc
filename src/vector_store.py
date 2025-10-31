import os

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
genai_embedding_model = os.getenv(
    "GOOGLE_GENAI_EMBEDDING_MODEL", "models/embedding-001"
)

if not genai_api_key:
    raise RuntimeError("GOOGLE_GENAI_API_KEY not set in environment or .env file")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=genai_embedding_model, google_api_key=SecretStr(genai_api_key)
)

# Create FAISS index
# Google's embedding-001 model produces 768-dimensional embeddings
# Using known dimension instead of API call to avoid quota usage during initialization
embedding_dimension = 768
index = faiss.IndexFlatL2(embedding_dimension)

# Initialize FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
