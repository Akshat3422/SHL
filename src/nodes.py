import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm
from src.state import AgentState, RetrievedDocument, Metadata, Entities
from src.retrieval import load_catalog, filter_catalog, search_bm25, search_vector
from src.prompts import (
    QUERY_UNDERSTANDING_SYSTEM,
    FINAL_ANSWER_SYSTEM,
    FINAL_ANSWER_HUMAN
)

logger = logging.getLogger(__name__)

_cross_encoder_model = None

def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        logger.info("Loading CrossEncoder model: cross-encoder/ms-marco-MiniLM-L-6-v2")
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder model loaded.")
    return _cross_encoder_model

# Define Pydantic schema for Structured Output in Query Understanding
class QueryAnalysis(BaseModel):
    off_topic: bool = Field(description="Set to true if the query is unrelated to SHL assessments, tests, skills, hiring products, catalog, or general assessment platform questions.")
    needs_clarification: bool = Field(description="Set to true if the query is too ambiguous, vague, or lacks essential detail to execute a catalog search, requiring clarification.")
    clarification_question: Optional[str] = Field(description="A friendly question asking the user for the necessary details if needs_clarification is True.")

    # Metadata filters
    category: List[str] = Field(default_factory=list, description="List of categories. Must match/map to: 'Biodata & Situational Judgment', 'Ability & Aptitude', 'Assessment Exercises', 'Simulations', 'Knowledge & Skills', 'Competencies', 'Personality & Behavior', 'Development & 360'.")
    job_levels: List[str] = Field(default_factory=list, description="Target job levels, e.g. 'Graduate', 'Manager', 'Executive', 'Professional'.")
    languages: List[str] = Field(default_factory=list, description="Languages requested, e.g. 'English', 'Spanish', 'French'.")
    remote: Optional[bool] = Field(None, description="Whether remote/online proctoring or access is requested.")
    adaptive: Optional[bool] = Field(None, description="Whether adaptive testing (test difficulty adjusts dynamically) is requested.")
    duration_max: Optional[int] = Field(None, description="Maximum test duration in minutes if specified.")

    # Entities
    assessments: List[str] = Field(default_factory=list, description="Specific names of assessments mentioned in the query.")

    # Refined Search Query
    semantic_query: str = Field(description="A clean semantic search query to use for vector/keyword search, removing conversational filler.")


def QueryUnderstandingFunction(state: AgentState) -> Dict[str, Any]:
    """Analyzes user query to extract filters and check intent, capping clarification at 2 attempts."""
    logger.info(f"[QueryUnderstanding] Processing query: {state['query']!r}")
    logger.info(f"[QueryUnderstanding] Current clarification_count: {state.get('clarification_count', 0)}")

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(QueryAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_UNDERSTANDING_SYSTEM),
        ("human", "{query}")
    ])
    
    # Build conversation history string
    history_str = ""
    for msg in state.get("chat_history", []):
        history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
    context_query = f"Conversation History:\n{history_str}\nCurrent User Input: {state['query']}" if history_str else state["query"]
    logger.info(f"[QueryUnderstanding] LLM Context:\n{context_query}")

    chain = prompt | structured_llm
    analysis: QueryAnalysis = chain.invoke({"query": context_query})

    current_count = state.get("clarification_count", 0)
    needs_clarification = analysis.needs_clarification
    clarification_question = analysis.clarification_question

    if needs_clarification:
        if current_count >= 2:
            logger.warning("[QueryUnderstanding] Clarification cap reached (2). Forcing search.")
            needs_clarification = False
            clarification_question = None
        else:
            current_count += 1
            logger.info(f"[QueryUnderstanding] Clarification needed ({current_count}/2): {clarification_question!r}")

    logger.info(
        f"[QueryUnderstanding] Result → off_topic={analysis.off_topic}, "
        f"needs_clarification={needs_clarification}, "
        f"categories={analysis.category}, job_levels={analysis.job_levels}, "
        f"semantic_query={analysis.semantic_query!r}"
    )

    return {
        "off_topic": analysis.off_topic,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "clarification_count": current_count,
        "metadata": {
            "category": analysis.category,
            "job_levels": analysis.job_levels,
            "languages": analysis.languages,
            "remote": analysis.remote,
            "adaptive": analysis.adaptive,
            "duration_max": analysis.duration_max,
        },
        "entities": {
            "assessments": analysis.assessments
        },
        "semantic_query": analysis.semantic_query
    }


def metaDataFiltering(state: AgentState) -> Dict[str, Any]:
    """Logs extracted metadata filters. Actual filtering happens inside each search node."""
    logger.info(f"[MetaDataFiltering] Extracted metadata: {state.get('metadata', {})}")
    logger.info(f"[MetaDataFiltering] Entities: {state.get('entities', {})}")
    return {}


def BM25Result(state: AgentState) -> Dict[str, Any]:
    """Performs BM25 search over the filtered catalog."""
    logger.info(f"[BM25] Running BM25 search for: {state['semantic_query']!r}")
    catalog = load_catalog()
    filtered_items = filter_catalog(catalog, state.get("metadata", {}))
    logger.info(f"[BM25] Catalog size: {len(catalog)}, After metadata filter: {len(filtered_items)}")
    bm25_res = search_bm25(state["semantic_query"], filtered_items, top_n=10)
    logger.info(f"[BM25] Retrieved {len(bm25_res)} results. Top: {[r['name'] for r in bm25_res[:3]]}")
    return {"bm25_results": bm25_res}


def Sementic_Search(state: AgentState) -> Dict[str, Any]:
    """Performs semantic vector search over the filtered catalog."""
    logger.info(f"[VectorSearch] Running semantic search for: {state['semantic_query']!r}")
    catalog = load_catalog()
    filtered_items = filter_catalog(catalog, state.get("metadata", {}))
    logger.info(f"[VectorSearch] Catalog size: {len(catalog)}, After metadata filter: {len(filtered_items)}")
    vector_res = search_vector(state["semantic_query"], filtered_items, top_n=10)
    logger.info(f"[VectorSearch] Retrieved {len(vector_res)} results. Top: {[r['name'] for r in vector_res[:3]]}")
    return {"vector_results": vector_res}


def RRF(state: AgentState) -> Dict[str, Any]:
    """Combines BM25 and Vector Search results using Reciprocal Rank Fusion."""
    bm25_res = state.get("bm25_results", [])
    vector_res = state.get("vector_results", [])
    logger.info(f"[RRF] Fusing {len(bm25_res)} BM25 results + {len(vector_res)} vector results")

    k = 60
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    for rank, doc in enumerate(bm25_res):
        eid = doc["entity_id"]
        doc_map[eid] = doc
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc in enumerate(vector_res):
        eid = doc["entity_id"]
        doc_map[eid] = doc
        rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank + 1)

    sorted_eids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for eid in sorted_eids:
        doc = doc_map[eid].copy()
        doc["score"] = rrf_scores[eid]
        doc["source"] = "rrf"
        fused_results.append(doc)

    logger.info(f"[RRF] Fused into {len(fused_results)} unique results. Top: {[d['name'] for d in fused_results[:3]]}")
    return {"fused_results": fused_results[:10]}


def Reranker(state: AgentState) -> Dict[str, Any]:
    """Uses CrossEncoder to rerank top-fused results for maximum semantic relevance."""
    fused = state.get("fused_results", [])
    logger.info(f"[Reranker] Reranking {len(fused)} fused results using CrossEncoder")

    if not fused:
        logger.warning("[Reranker] No fused results to rerank.")
        return {"reranked_results": []}

    model = get_cross_encoder()
    pairs = [(state["query"], doc.get("description", "")) for doc in fused]
    scores = model.predict(pairs)

    scored_docs = list(zip(scores, fused))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    reranked = [doc for score, doc in scored_docs]
    logger.info(f"[Reranker] Top reranked results: {[d['name'] for d in reranked[:3]]}")
    return {"reranked_results": reranked}


def clarfying_question(state: AgentState) -> Dict[str, Any]:
    """Returns the clarification question as the final_answer so the API can surface it."""
    q = state.get("clarification_question", "Could you please specify which assessment category or job role you are looking for?")
    logger.info(f"[ClarifyingQuestion] Asking: {q!r}")
    return {"final_answer": q}


def final_answer(state: AgentState) -> Dict[str, Any]:
    """Synthesizes final answer from reranked search results using the LLM."""
    reranked = state.get("reranked_results", [])
    logger.info(f"[FinalAnswer] Generating answer from {len(reranked)} reranked results")

    if not reranked:
        logger.warning("[FinalAnswer] No reranked results available.")
        return {"final_answer": "I couldn't find any assessments matching your query in the catalog. Please try a different query or specify different filters."}

    llm = get_llm(temperature=0.0)

    items_str = ""
    for doc in reranked[:5]:
        items_str += f"**{doc['name']}**\n- Link: {doc['link']}\n- Description: {doc['description']}\n\n"

    user_prompt = FINAL_ANSWER_HUMAN.format(
        query=state["query"],
        matching_assessments=items_str
    )

    response = llm.invoke([
        ("system", FINAL_ANSWER_SYSTEM),
        ("human", user_prompt)
    ])

    logger.info("[FinalAnswer] Answer generated successfully.")
    return {"final_answer": response.content}
