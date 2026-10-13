import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")  # vérifie le nom actuel dans la console Groq

# Document en anglais : bge-small-en. Document en français :
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (aussi 384 dimensions)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

PDF_PATH = os.getenv("PDF_PATH", "data/document.pdf")
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_data")
COLLECTION = "docs"

CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))