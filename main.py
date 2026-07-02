import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.graph import create_graph

def main():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable is not set. Please set it in a .env file or your shell environment.")
        
    print("Initializing SHL Product Catalog Search Agent...")
    graph = create_graph()
    print("Agent Initialized. Type 'exit' to quit.\n")
    
    # Session config for checkpointing
    config = {"configurable": {"thread_id": "shl-search-session"}}
    
    while True:
        try:
            query = input("Ask about assessments: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
                
            print("\nProcessing request...")
            initial_state = {
                "query": query,
                "off_topic": False,
                "needs_clarification": False,
                "clarification_question": None,
                "clarification_count": 0,
                "metadata": {},
                "entities": {},
                "semantic_query": "",
                "bm25_results": [],
                "vector_results": [],
                "fused_results": [],
                "reranked_results": [],
                "final_answer": ""
            }
            
            # Invoke the graph passing the thread config
            output = graph.invoke(initial_state, config=config)
            
            print("\n--- Agent Response ---")
            if output.get("off_topic"):
                print("Sorry, this query is off-topic. Please ask questions about SHL assessments and tests.")
            else:
                print(output.get("final_answer", "No response generated."))
            print("----------------------\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during query execution: {e}")

if __name__ == "__main__":
    main()
