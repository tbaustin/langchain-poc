import logging
import os
import sqlite3

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .nodes import (
    ask_user_for_feedback,
    call_model,
    check_end_connection,
    check_feedback,
    get_user_input,
    provide_feedback,
)
from .state import GraphState

logger = logging.getLogger(__name__)


def create_graph():
    """Builds and compiles the LangGraph."""

    # Init the correct checkpointer based on env
    if os.getenv("ENV") == "production":
        conn = sqlite3.connect("memory.sqlite")
        memory = SqliteSaver(conn=conn)
        logger.info("Using SqliteSaver for production.")
    else:
        memory = InMemorySaver()
        logger.info("Using InMemorySaver for development.")

    graph = (
        StateGraph(GraphState)
        .add_node("call_model", call_model)
        .add_node("get_user_input", get_user_input)
        .add_node("provide_feedback", provide_feedback)
        .add_node("ask_user_for_feedback", ask_user_for_feedback)
        .set_entry_point("get_user_input")
        .add_conditional_edges(
            "get_user_input",
            check_end_connection,
            {"end": END, "continue": "call_model"},
        )
        .add_edge("call_model", "ask_user_for_feedback")
        .add_conditional_edges(
            "ask_user_for_feedback",
            check_feedback,
            {"feedback": "provide_feedback", "continue": "get_user_input"},
        )
        .add_edge("provide_feedback", "call_model")
        .compile(checkpointer=memory)
    )

    logger.info("Graph initialized successfully")
    return graph
