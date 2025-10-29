import logging
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from .chains import create_genai_chain
from .state import GraphState
from .vector_store import vector_store

# Configure logging
logger = logging.getLogger(__name__)

# Constants
END_COMMANDS = ["quit", "exit", "end", "no"]


def retrieve_feedback(state: GraphState):
    """
    Retrieves the most relevant feedback from the long-term memory.
    """
    logger.info("Retrieving relevant feedback from vector store...")
    last_user_message = state.messages[-1].content
    relevant_docs = vector_store.similarity_search(last_user_message, k=1)

    retrieved_feedback = (
        relevant_docs[0].page_content
        if relevant_docs
        else "No relevant feedback found."
    )
    logger.info(f"Retrieved feedback: {retrieved_feedback}")
    return {"retrieved_feedback": retrieved_feedback}


def save_feedback(state: GraphState):
    """
    Saves the user's feedback to the long-term memory.
    """
    logger.info("Saving user feedback to vector store...")

    # Get the feedback from the last message
    feedback_text = state.messages[-1].content

    # Get recipe context from the 2nd to last message (should be the recipe response)
    recipe_context = None
    if len(state.messages) >= 2:
        second_to_last = state.messages[-2]
        if (
            hasattr(second_to_last, "content")
            and "recipe" in second_to_last.content.lower()
        ):
            recipe_context = second_to_last.content[:200]  # First 200 chars as context
            logger.info(f"Recipe context extracted: {recipe_context[:30]}...")

    # Create a Document with feedback and metadata
    feedback_doc = Document(
        page_content=feedback_text,
        metadata={
            "type": "user_feedback",
            "timestamp": datetime.now().isoformat(),
            "recipe_context": recipe_context,
        },
    )

    # Add to vector store
    vector_store.add_documents([feedback_doc])
    logger.info("User feedback saved successfully")

    return state


def call_model(state: GraphState):
    """
    Calls the LLM with the user's query and the retrieved feedback.
    """
    logger.info("Processing user request with context...")

    ai_response = create_genai_chain().invoke(
        {"messages": state.messages, "retrieved_feedback": state.retrieved_feedback}
    )
    logger.info("AI response generated successfully")
    return {"messages": [ai_response]}


def check_end_connection(state: GraphState):
    logger.info("Checking for end condition")
    try:
        last_content = state.messages[-1].content
        result = "end" if last_content in END_COMMANDS else "continue"
        logger.info(f"User input: '{last_content}', routing decision: {result}")
        return result
    except AttributeError as e:
        logger.warning(f"Error accessing last_content: {e}, defaulting to 'continue'")
        return "continue"


def get_user_input(state: GraphState):
    user_input = input("Please provide your sourdough recipe request: ")
    return {
        "messages": [HumanMessage(content=user_input)],
    }


def provide_feedback(state: GraphState):
    feedback = input("Please provide your feedback on the recipe: ")
    return {
        "messages": [HumanMessage(content=feedback)],
    }


def ask_user_for_feedback(state: GraphState):
    user_input = (
        input("Do you want to provide feedback on the recipe? (yes/no): ")
        .strip()
        .lower()
    )
    return {
        "messages": [HumanMessage(content=user_input)],
    }


def check_feedback(state: GraphState):
    last_message = state.messages[-1]
    if last_message.content in ["yes", "y"]:
        return "feedback"
    return "continue"
