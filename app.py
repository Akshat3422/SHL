import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.graph import create_graph
from src.retrieval import load_catalog

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Product Catalog Search API")

# Initialize graph once at startup
graph = create_graph()

# Load and map catalog for quick lookup of keys -> test_type
try:
    catalog = load_catalog()
    catalog_map = {item.get("entity_id"): item for item in catalog if item.get("entity_id")}
except Exception as e:
    logger.error(f"Error loading catalog: {e}")
    catalog_map = {}

# Category mapping to test_type
TEST_TYPE_MAP = {
    'ability & aptitude': 'A',
    'assessment exercises': 'E',
    'biodata & situational judgment': 'B',
    'competencies': 'C',
    'development & 360': 'D',
    'knowledge & skills': 'K',
    'personality & behavior': 'P',
    'simulations': 'S'
}

def get_test_type(keys: List[str]) -> str:
    if not keys:
        return "O"
    for key in keys:
        mapped = TEST_TYPE_MAP.get(key.strip().lower())
        if mapped:
            return mapped
    return "O"

# ── Request / Response Models (non-negotiable schema) ──

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = None
    # Backward compatibility: also accept a single query/message string
    message: Optional[str] = None
    query: Optional[str] = None
    session_id: Optional[str] = None

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # ── Parse the incoming messages ──
    if request.messages and len(request.messages) > 0:
        # Stateless mode: full conversation history provided
        chat_history = [{"role": m.role, "content": m.content} for m in request.messages]
        # Last user message is the current query
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found in messages array.")
        query_text = user_messages[-1].content.strip()
        # Build chat_history (everything except the last user message)
        history_for_state = chat_history[:-1]  # All messages before the latest
    else:
        # Backward compatibility: single query string
        query_text = request.message or request.query
        if not query_text or not query_text.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        query_text = query_text.strip()
        history_for_state = []

    # Count how many clarification rounds already happened (assistant messages in history)
    clarification_count = sum(1 for m in history_for_state if m.get("role") == "assistant")

    logger.info(f"[/chat] query={query_text!r} | history_turns={len(history_for_state)} | clarification_count={clarification_count}")

    try:
        # Build fresh state for every request (stateless)
        invoke_state = {
            "query": query_text,
            "chat_history": history_for_state,
            "off_topic": False,
            "needs_clarification": False,
            "clarification_question": None,
            "clarification_count": clarification_count,
            "metadata": {},
            "entities": {},
            "semantic_query": "",
            "bm25_results": [],
            "vector_results": [],
            "fused_results": [],
            "reranked_results": [],
            "final_answer": ""
        }

        output = graph.invoke(invoke_state)

        # 1. Handle Off Topic
        if output.get("off_topic"):
            return ChatResponse(
                reply="Sorry, this query is off-topic. Please ask questions about SHL assessments and tests.",
                recommendations=[],
                end_of_conversation=False
            )
            
        # 2. Handle Clarification Needed
        if output.get("needs_clarification"):
            reply_text = output.get("final_answer") or output.get("clarification_question") or "Could you please clarify your request?"
            return ChatResponse(
                reply=reply_text,
                recommendations=[],
                end_of_conversation=False
            )
            
        # 3. Handle Successful Search / Recommendations
        reranked = output.get("reranked_results", [])
        if reranked:
            recs = []
            for doc in reranked[:10]:
                eid = doc.get("entity_id")
                catalog_item = catalog_map.get(eid, {})
                keys = catalog_item.get("keys", [])
                test_type = get_test_type(keys)
                
                recs.append(Recommendation(
                    name=doc.get("name", ""),
                    url=doc.get("link", ""),
                    test_type=test_type
                ))
                
            return ChatResponse(
                reply=output.get("final_answer", ""),
                recommendations=recs,
                end_of_conversation=True
            )
            
        # 4. Default Fallback (Empty Results)
        return ChatResponse(
            reply=output.get("final_answer") or "I couldn't find any matching assessments in the catalog.",
            recommendations=[],
            end_of_conversation=True
        )

    except Exception as e:
        logger.error(f"[/chat] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=ChatResponse)
def query_endpoint(request: ChatRequest):
    return chat(request)

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000 when run directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
