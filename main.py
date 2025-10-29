import logging

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.graph import create_graph
from src.state import GraphState

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

graph = create_graph()


def main():
    logger.info("Starting application")
    print("Starting chat with Google Generative AI...")

    thread_id = "recipe-assistant-1"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Check if there's existing conversation history for this thread
    try:
        existing_state = graph.get_state(config)
        if existing_state.values:
            input_state = None
        else:
            input_state = GraphState()
    except Exception:
        input_state = GraphState()

    for step in graph.stream(input_state, config):
        for node_name, node_output in step.items():
            print(f"\nNode '{node_name}' executed:")

            if isinstance(node_output, dict) and "messages" in node_output:
                last_message = node_output["messages"][-1]
                if hasattr(last_message, "content"):
                    print(f"  Content: {last_message.content}")
            else:
                print(f"  Output: {node_output}")


if __name__ == "__main__":
    main()
