from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "memory"] = Field(
        ...,
        description="Given a user question choose to route it to memory or a vectorstore.",
    )

def route(llm, memory):
    structured_llm_router = llm.with_structured_output(RouteQuery)
    memory_str = "\n".join(memory)
    system = f"""The user has sent a new message. You are an expert at routing the message.
     Based on the content of the message, determine the next step by following these guidelines:
     Firstly take a deep breath and read the message carefully.
     If it is not english, convert it to english first, so u can better understand.
     If user asking about someone, firstly check the vectorstore.
     The question might look like a general question, but it may be about a specific document in the vectorstore.
     Don't mistake between the vectorstore and memory, question may be about a document in the vectorstore.
     If the message requires more detailed or additional information that may be found in the stored documents within the vector database, return vectorstore.
     If the message references previous parts of the conversation or context that has already been discussed, or just daily talk, return memory.

    Conversation history:
    {memory_str}
    """


    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    return route_prompt | structured_llm_router