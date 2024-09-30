from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import TypedDict, List
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from router import route, RouteQuery
from ingestion import PDFIngestor


class GraphState(TypedDict):
    question: str
    response: str
    documents: List[str]


class PdfChat:
    def __init__(self, api_key, retriever):
        self.model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini-2024-07-18", temperature=0)  # minimum costing model
        builder = StateGraph(GraphState)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("boost_retrieve", self.boost_retrieve)
        builder.add_node("generate_with_rag", self.generate_with_doc)
        builder.add_node("generate", self.generate_wo_doc)
        builder.set_conditional_entry_point(
            self.decide_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )
        builder.add_edge("retrieve", "boost_retrieve")
        builder.add_edge("retrieve", "generate_with_rag")
        builder.add_edge("generate_with_rag", END)
        builder.add_edge("generate", END)

        self.retriever = retriever

        self.graph = builder.compile()

        self.memory = ConversationBufferMemory()

    def decide_retrieve(self, state: GraphState):
        question = state["question"]
        memory = self.memory.load_memory_variables({})
        source: RouteQuery = route(self.model, memory).invoke({"question": question})
        if source.datasource == "vectorstore":
            return "retrieve"
        else:
            return "generate"

    def boost_retrieve(self, state: GraphState):
        question = state["question"]

        prompt = """You are an assistant in a question-answering tasks.
                    You have to boost the question to help search in vectorstore.
                    Don't make up random names.
                    Return a better structred question for vectorstore search, but don't make it longer
                    \n
                    Conversation history: {memory}
                    \n
                    Question: {question}
                """
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | self.model | StrOutputParser()

        question = chain.invoke({"question": question, "memory": self.memory.load_memory_variables({})})
        print("Boosted question:", question)
        return {"question": question}

    def retrieve_node(self, state: GraphState):
        question = state["question"]

        documents = self.retriever.invoke(question)
        if not documents:
            return "I couldn't find any relevant documents. Can you please rephrase your question?"

        return {"documents": documents}

    def generate_with_doc(self, state: GraphState):
        documents = state["documents"]
        question = state["question"]
        memory = self.memory.load_memory_variables({})

        prompt = """"You are an expert assistant for question-answering tasks. 
                    Use the provided documents as context to extract and answer the question. 
                    Read the context carefully, processing each document individually. 
                    Pay close attention to the source of the information—ensure that your answer is specific to the correct document, and do not mix details between documents. 
                    Focus on structured data like phone numbers, addresses, email addresses, or URLs, and extract them exactly as provided. 
                    If such information is hidden within the context, identify and use it. 
                    If the answer is not present, respond with 'I don't know.' 
                    Keep your answer concise and limited to three sentences.

                    Conversation history: {memory}
                    Context: {context}
                    Question: {question} 
                    Answer:
                """
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | self.model | StrOutputParser()

        response = chain.invoke({"memory": memory, "question": question, "context": documents})

        self.memory.save_context(inputs={"input": question}, outputs={"output": response})

        return {"response": response}

    def generate_wo_doc(self, state: GraphState):
        question = state["question"]
        prompt = """You are an assistant for question-answering tasks. 
                    If you don't know the answer, just say that you don't know.
                    Don't forget to check previous conversations for context.
                    Use three sentences maximum and keep the answer concise. 
                    Conversation history: {memory} 
                    Question: {question}
                """
        memory = self.memory.load_memory_variables({})
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | self.model | StrOutputParser()

        response = chain.invoke({"memory": memory, "question": question})
        self.memory.save_context(inputs={"input": question}, outputs={"output": response})

        return {"response": response}
