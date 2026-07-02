from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.nodes import (
    QueryUnderstandingFunction,
    metaDataFiltering,
    BM25Result,
    clarfying_question,
    Sementic_Search,
    RRF,
    Reranker,
    final_answer
)

def create_graph():
    builder = StateGraph(AgentState)
    
    # Register Nodes
    builder.add_node("QueryUnderstanding", QueryUnderstandingFunction)
    builder.add_node("metaDataFiltering", metaDataFiltering)
    builder.add_node("BM25Result", BM25Result)
    builder.add_node("clarfying_question", clarfying_question)
    builder.add_node("Sementic_Search", Sementic_Search)
    builder.add_node("RRF", RRF)
    builder.add_node("Reranker", Reranker)
    builder.add_node("final_answer", final_answer)
    
    # Add Edges
    builder.add_edge(START, "QueryUnderstanding")
    
    # Conditional Edges for Router
    builder.add_conditional_edges(
        "QueryUnderstanding",
        lambda state: (
            "off_topic" if state.get("off_topic")
            else "clarification" if state.get("needs_clarification")
            else "metadata"
        ),
        {
            "off_topic": END,
            "clarification": "clarfying_question",
            "metadata": "metaDataFiltering",
        },
    )
    
    builder.add_edge("clarfying_question", END)
    builder.add_edge("metaDataFiltering", "BM25Result")
    builder.add_edge("metaDataFiltering", "Sementic_Search")
    builder.add_edge("BM25Result", "RRF")
    builder.add_edge("Sementic_Search", "RRF")
    builder.add_edge("RRF", "Reranker")
    builder.add_edge("Reranker", "final_answer")
    builder.add_edge("final_answer", END)
    
    # Stateless — no checkpointer. The client sends full history every request.
    return builder.compile()
