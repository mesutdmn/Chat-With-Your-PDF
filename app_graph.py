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
        builder.add_node("boost_question", self.boost_question)
        builder.add_node("structer_document", self.structer_document)
        builder.add_node("generate_with_rag", self.generate_with_doc)
        builder.add_node("generate", self.generate_wo_doc)

        builder.set_entry_point("boost_question")
        builder.add_conditional_edges(
            "boost_question",
            self.decide_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )
        builder.add_edge("retrieve", "structer_document")
        builder.add_edge("structer_document", "generate_with_rag")
        builder.add_edge("generate_with_rag", END)
        builder.add_edge("generate", END)

        self.retriever = retriever

        self.graph = builder.compile()

        self.graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
        self.memory = ConversationBufferMemory()

    def decide_retrieve(self, state: GraphState):
        question = state["question"]
        memory = self.memory.load_memory_variables({})
        source: RouteQuery = route(self.model, memory).invoke({"question": question})
        if source.datasource == "vectorstore":
            return "retrieve"
        else:
            return "generate"

    def boost_question(self, state: GraphState):
        question = state["question"]
        memory = self.memory.load_memory_variables({})
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

        question = chain.invoke({"question": question, "memory": memory})
        print("Boosted question:", question)
        return {"question": question}


    def retrieve_node(self, state: GraphState):
        question = state["question"]

        documents = self.retriever.invoke(question)
        if not documents:
            return "I couldn't find any relevant documents. Can you please rephrase your question?"

        return {"documents": documents}

    def structer_document(self, state: GraphState):
        documents = state["documents"]
        question = state["question"]
        documents = [doc.page_content for doc in documents]

        prompt = """You are an expert assistant for question-answering tasks. 
                    You have to restructure the documents for the question. 
                    Keep it short, only knowledge that is relevant to the question.
                    Don't make up random names.
                    Return a better structured document for better understanding.
                    \n
                    Documents: {documents}
                    \n
                    Question: {question}
                """
        prompt = PromptTemplate.from_template(prompt)
        chain = prompt | self.model | StrOutputParser()

        document = chain.invoke({"question": question, "documents": documents})

        print(document)
        return  {"documents": document}

    def generate_with_doc(self, state: GraphState):
        documents = state["documents"]
        question = state["question"]
        memory = self.memory.load_memory_variables({})

        prompt = """"You are an expert assistant for question-answering tasks. 
                    Use the provided documents as context to extract and answer the question. 

                    If the answer is not mentioned in context, respond with 'I don't know.' 
                    Keep your limited to three sentences.

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
