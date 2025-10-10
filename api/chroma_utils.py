# api/chroma_utils.py
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import traceback

# Try to create a robust embedding function:
try:
    # Preferred: LangChain HuggingFace wrapper (may try to download model)
    from langchain_huggingface import HuggingFaceEmbeddings
    def _make_embedding_fn():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_function = _make_embedding_fn()
except Exception:
    # Fallback: use sentence-transformers directly (works offline if model cached)
    try:
        from sentence_transformers import SentenceTransformer
        class _SentenceTransformerWrapper:
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            def embed_documents(self, texts):
                # expects list[str] -> list[list[float]]
                return [self.model.encode(t, show_progress_bar=False).tolist() for t in texts]
            def embed_query(self, text):
                return self.model.encode(text).tolist()
        embedding_function = _SentenceTransformerWrapper("all-MiniLM-L6-v2")
    except Exception:
        # If both fail, raise a helpful error at import time (you may see it locally)
        raise RuntimeError("Failed to create any embedding function. Ensure sentence-transformers or langchain_huggingface is installed and internet access is available to download models if not cached.")

# Ensure chroma directory exists
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "./chroma_db")
os.makedirs(CHROMA_DIR, exist_ok=True)

from langchain_chroma import Chroma
# Create vectorstore (persistent)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_function)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

def load_and_split_document(file_path: str):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """
    Splits a document, attaches metadata.file_id and indexes into Chroma.
    Persists the DB after adding.
    """
    try:
        splits = load_and_split_document(file_path)
        for split in splits:
            if not getattr(split, "metadata", None):
                split.metadata = {}
            split.metadata['file_id'] = file_id
        vectorstore.add_documents(splits)
        try:
            vectorstore.persist()
        except Exception:
            # persist may not be necessary for some Chroma builds; ignore if fails
            pass
        return True
    except Exception as e:
        print("Error indexing document:", e)
        traceback.print_exc()
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        # Chroma Python client supports delete via where argument
        # Try the high-level API if available
        try:
            vectorstore._collection.delete(where={"file_id": file_id})
        except Exception:
            # Fallback: try vectorstore.delete or similar interfaces
            try:
                vectorstore.delete(where={"file_id": file_id})
            except Exception as e:
                print("Delete fallback failed:", e)
                raise
        try:
            vectorstore.persist()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {e}")
        return False
