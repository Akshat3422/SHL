import os
import json
import requests
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import CATALOG_URL, CATALOG_CACHE_PATH, FAISS_INDEX_PATH
from src.state import RetrievedDocument

# Custom Headers to fetch catalog
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.shl.com/"
}

def load_catalog() -> List[Dict[str, Any]]:
    """Loads the catalog from the local cache, or downloads it if cache doesn't exist."""
    if os.path.exists(CATALOG_CACHE_PATH):
        with open(CATALOG_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # Download catalog
    response = requests.get(CATALOG_URL, headers=HEADERS)
    response.raise_for_status()
    text = response.text.replace(
        '"Microsoft \n    365 (New)"',
        '"Microsoft 365 (New)"'
    )
    data = json.loads(text)
    
    # Save cache
    with open(CATALOG_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
    return data

def filter_catalog(catalog: List[Dict[str, Any]], metadata_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filters the catalog items based on categories, job levels, languages, adaptive, remote, duration."""
    filtered = []
    for item in catalog:
        # 1. Filter by categories — skip if list is empty (no filter)
        if metadata_filters.get("category"):
            item_keys = [k.lower() for k in item.get("keys", [])]
            # Bidirectional substring match: filter term in catalog key OR catalog key in filter term
            if not any(
                cat.lower() in k or k in cat.lower()
                for cat in metadata_filters["category"]
                for k in item_keys
            ):
                continue

        # 2. Filter by job levels — skip if list is empty (no filter)
        if metadata_filters.get("job_levels"):
            item_levels = [l.lower() for l in item.get("job_levels", [])]
            # Bidirectional: "Professional" matches "Professional Individual Contributor" and vice versa
            if not any(
                lvl.lower() in l or l in lvl.lower()
                for lvl in metadata_filters["job_levels"]
                for l in item_levels
            ):
                continue

        # 3. Filter by languages — skip if list is empty (no filter)
        if metadata_filters.get("languages"):
            item_langs = [l.lower() for l in item.get("languages", [])]
            if not any(
                lang.lower() in l or l in lang.lower()
                for lang in metadata_filters["languages"]
                for l in item_langs
            ):
                continue

        # 4. Filter by adaptive
        if metadata_filters.get("adaptive") is not None:
            val = metadata_filters["adaptive"]
            catalog_val = item.get("adaptive", "").lower()
            if val is True and catalog_val != "yes":
                continue
            if val is False and catalog_val == "yes":
                continue

        # 5. Filter by remote
        if metadata_filters.get("remote") is not None:
            val = metadata_filters["remote"]
            catalog_val = item.get("remote", "").lower()
            if val is True and catalog_val != "yes":
                continue
            if val is False and catalog_val == "yes":
                continue

        # 6. Filter by duration_max
        if metadata_filters.get("duration_max") is not None:
            try:
                duration_str = item.get("duration", "0")
                duration = int(''.join(filter(str.isdigit, duration_str)) or 0)
                if duration > 0 and duration > metadata_filters["duration_max"]:
                    continue
            except Exception:
                pass

        filtered.append(item)
    return filtered

def search_bm25(query: str, items: List[Dict[str, Any]], top_n: int = 10) -> List[RetrievedDocument]:
    """Runs BM25 search on a list of filtered catalog items."""
    if not items:
        return []
    
    # Tokenize descriptions / names for BM25
    corpus = []
    for item in items:
        text_content = f"{item.get('name', '')} {item.get('description', '')} " + " ".join(item.get('keys', []))
        corpus.append(text_content.lower().split())
        
    bm25 = BM25Okapi(corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    results = []
    for idx, score in enumerate(scores):
        if score > 0:
            item = items[idx]
            results.append({
                "entity_id": item.get("entity_id", f"idx_{idx}"),
                "name": item.get("name", ""),
                "link": item.get("link", ""),
                "description": item.get("description", ""),
                "score": float(score),
                "source": "bm25",
                "keys": item.get("keys", [])
            })
            
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

def search_vector(query: str, items: List[Dict[str, Any]], top_n: int = 10) -> List[RetrievedDocument]:
    """Runs Vector Search using FAISS and HuggingFaceEmbeddings on filtered catalog items."""
    if not items:
        return []
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    documents = []
    for idx, item in enumerate(items):
        text_content = f"Name: {item.get('name', '')}\nDescription: {item.get('description', '')}\nCategories: {', '.join(item.get('keys', []))}"
        doc = Document(
            page_content=text_content,
            metadata={
                "entity_id": item.get("entity_id", f"idx_{idx}"),
                "name": item.get("name", ""),
                "link": item.get("link", ""),
                "keys": item.get("keys", [])
            }
        )
        documents.append(doc)
        
    db = FAISS.from_documents(documents, embeddings)
    search_results = db.similarity_search_with_score(query, k=top_n)
    
    results = []
    for doc, distance in search_results:
        similarity = 1.0 / (1.0 + float(distance))
        results.append({
            "entity_id": doc.metadata.get("entity_id"),
            "name": doc.metadata.get("name"),
            "link": doc.metadata.get("link"),
            "description": doc.page_content,
            "score": similarity,
            "source": "vector",
            "keys": doc.metadata.get("keys", [])
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
