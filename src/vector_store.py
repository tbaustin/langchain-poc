import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
genai_embedding_model = os.getenv(
    "GOOGLE_GENAI_EMBEDDING_MODEL", "models/embedding-001"
)

if not genai_api_key:
    raise RuntimeError("GOOGLE_GENAI_API_KEY not set in environment or .env file")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=genai_embedding_model)

# Create FAISS index
# Get the embedding dimension by calling the model with a test query
# FAISS requires knowing the vector dimension upfront to create the index
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

# Initialize FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
