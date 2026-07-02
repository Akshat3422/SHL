import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Catalog URL
CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

# Local cache paths - Vercel Serverless Functions have read-only file systems except for /tmp
if os.environ.get("VERCEL"):
    DATA_DIR = "/tmp/shl_data"
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

os.makedirs(DATA_DIR, exist_ok=True)
CATALOG_CACHE_PATH = os.path.join(DATA_DIR, "shl_product_catalog.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index")

GROQ_MODEL_NAME = "openai/gpt-oss-120b"

def get_llm(temperature: float = 0.0):
    return ChatGroq(
        temperature=temperature,
        model_name=GROQ_MODEL_NAME
    )
