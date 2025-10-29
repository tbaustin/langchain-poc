from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class GraphState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    retrieved_feedback: str = Field(default="")
