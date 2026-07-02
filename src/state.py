from typing import TypedDict, Optional, List, Dict, Annotated
import operator

class Metadata(TypedDict, total=False):
    category: List[str]
    job_levels: List[str]
    languages: List[str]
    remote: Optional[bool]
    adaptive: Optional[bool]
    duration_max: Optional[int]

class Entities(TypedDict, total=False):
    assessments: List[str]

class RetrievedDocument(TypedDict):
    entity_id: str
    name: str
    link: str
    description: str
    score: float
    source: str   # "bm25", "vector", "rrf"

class AgentState(TypedDict):
    # Input
    query: str
    chat_history: Annotated[list, operator.add]

    # Query Understanding
    off_topic: bool
    needs_clarification: bool
    clarification_question: Optional[str]
    clarification_count: int  # Track number of clarifications asked

    # Parsed Query
    metadata: Metadata
    entities: Entities
    semantic_query: str

    # Retrieval
    bm25_results: List[RetrievedDocument]
    vector_results: List[RetrievedDocument]
    fused_results: List[RetrievedDocument]
    reranked_results: List[RetrievedDocument]

    # Output
    final_answer: str
