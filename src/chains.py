import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr


def create_genai_chain():
    """
    Creates and returns a LangChain conversational chain for sourdough recipes.
    """
    genai_api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    genai_model = os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash-lite")

    if not genai_api_key:
        raise RuntimeError("GOOGLE_GENAI_API_KEY not set in environment or .env file")

    genai_llm = ChatGoogleGenerativeAI(
        model=genai_model,
        api_key=SecretStr(genai_api_key),
        temperature=0.7,
    )

    prompt_template = """
    Context:
        - Use only sourdough recipes
        - Put the unit measurements in grams
        - Convert butter to margarine
        - Convert honey to granulated sugar
        - Preference is to have soft, not chewy sourdough
        - Keep the recipe to 1000 characters or less.
        - Always start your response with "Here's a sourdough recipe:" when providing a recipe

        You are a helpful assistant that provides sourdough recipes based on user requests and
        feedback. When the user provides feedback, adjust the recipe to meet their needs.

        You have access to a long-term memory of the user's past feedback on your recipes. 
        Use this feedback to personalize your new recipes to better suit the user's preferences.

        User Feedback Context: {retrieved_feedback}
    """

    chat_prompt = ChatPromptTemplate(
        [("system", prompt_template), MessagesPlaceholder(variable_name="messages")]
    )

    genai_chain = chat_prompt | genai_llm
    return genai_chain
