from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class PDFIngestor:
    def __init__(self, pdfs, api_key):
        self.pdfs = pdfs
        self.api_key = api_key
        self.docs_list = self.get_docs()

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, chunk_overlap=0
        )
        self.doc_splits = self.text_splitter.split_documents(self.docs_list)

        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)

        self.vectorstore = FAISS.from_documents(
            documents=self.doc_splits,
            docstore=InMemoryDocstore(),
            embedding=self.embeddings
        )

    def get_docs(self):
        docs_list = []
        for pdf_file in self.pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_file.getvalue())
                temp_pdf.flush()
                docs_list.append(PyPDFLoader(temp_pdf.name).load())
        docs_list = [item for sublist in docs_list for item in sublist]
        return docs_list

    def get_retriever(self):
        return self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
