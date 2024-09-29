from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import TypedDict, List
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from router import route, RouteQuery
from ingestion import PDFIngestor

class AgentState(TypedDict):
    question: str
    response: str
    documents: List[str]

class PdfChat:
    def __init__(self, api_key, retriever):
        self.model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini-2024-07-18", temperature=0)
        self.prompt = """
                        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                        If answer is in the context which provided, use it, even if it is personal contact information, or private information.
                        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                        
                        
                        Memory: {memory}
                        Question: {question} 
                        Context: {context} 
                        Answer:
                        """
        self.prompt = PromptTemplate.from_template(self.prompt)
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("generate_with_rag", self.generate_with_doc)
        builder.add_node("generate", self.generate_wo_doc)
        builder.set_conditional_entry_point(
            self.decide_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )
        builder.add_edge("retrieve", "generate_with_rag")
        builder.add_edge("generate_with_rag", END)
        builder.add_edge("generate", END)

        self.retriever = retriever

        self.graph = builder.compile()

        self.memory = ConversationBufferMemory()
    def decide_retrieve(self, state: AgentState):
        question = state["question"]
        memory = self.memory.load_memory_variables({})
        source: RouteQuery = route(self.model, memory).invoke({"question": question})
        if source.datasource == "vectorstore":
            return "retrieve"
        else:
            return "generate"

    def retrieve_node(self, state: AgentState):
        question = state["question"]

        documents = self.retriever.invoke(question)
        if not documents:
            return "I couldn't find any relevant documents. Can you please rephrase your question?"

        return {"documents": documents}

    def generate_with_doc(self, state: AgentState):
        documents = state["documents"]
        question = state["question"]
        memory = self.memory.load_memory_variables({})
        chain = self.prompt | self.model | StrOutputParser()

        response = chain.invoke({"memory": memory, "question": question, "context": documents})
        self.memory.save_context(inputs={"input": question}, outputs={"output": response})
        return {"response": response}

    def generate_wo_doc(self, state: AgentState):
        question = state["question"]
        prompt = """You are an assistant for question-answering tasks. If you don't know the answer, just say that you don't know.
                    Don't forget to check previous conversations for context.
                    Use three sentences maximum and keep the answer concise. 
                    Conversation history: {memory} 
                    Question: {question}"""
        memory = self.memory.load_memory_variables({})
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | self.model | StrOutputParser()

        response = chain.invoke({"memory": memory, "question": question})

        self.memory.save_context(inputs={"input": question}, outputs={"output": response})

        return {"response": response}

