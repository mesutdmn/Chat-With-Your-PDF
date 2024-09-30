
![cover](https://github.com/user-attachments/assets/6b66d12b-b3b6-4ffa-9ac3-8d31fec295e5)

## RAG PDF Chatbot

This project is a **Retrieval-Augmented Generation (RAG)**-based AI chatbot, capable of querying PDF documents using the **OpenAI API** integrated with **LangChain**. With **LangGraph** and **FAISS**, vector-based data queries are performed on PDF files, allowing natural language interaction with the data.

🤖 **Live Demo**: [RAG PDF Chatbot](https://rag-chat-pdf.streamlit.app/)

⚙️ **How it Works:**
![how-it-works](https://github.com/user-attachments/assets/c5a11c23-f9d8-4d96-a4e3-d8f49d6d77c7)


📂 **Project Structure:**
- **LangChain**: LangChain is used to process the information within the PDF files and pass it to the language model. The user's questions are enriched with relevant sections from the PDF to produce more accurate answers.
- **Vectorization**: FAISS is used to vectorize the PDF files, allowing for efficient and accurate data retrieval.
- **OPENAI API**: The OpenAI API is used to generate responses to the user's questions based on the information extracted from the PDF files.
- **LangGraph**: LangGraph is used to generate the graph structure of the chain, which is then used to enrich the user's questions with relevant information from the PDF files.

🕸️ **Graph Map**
![graph_output](https://github.com/user-attachments/assets/0fea4277-414d-40c0-a9e3-0cbc0416a0e9)

🎯 **Use Cases:**
- **Research**: Quickly find relevant information from research papers and articles.
- **Education**: Get answers to questions from textbooks and study materials.
- **Business**: Extract data from reports and documents for analysis and decision-making.
- **Legal**: Search for specific information in legal documents and contracts.
- **Healthcare**: Retrieve information from medical journals and reports.
- **Finance**: Extract data from financial reports and documents.
- **Customer Support**: Provide quick and accurate answers to customer queries.
- **General Knowledge**: Get answers to general questions from a wide range of sources.
- **And more...**

📦 **Installation:**
1. Clone the repository:
   ```bash
   git clone https://github.com/mesutdmn/Chat-With-Your-PDF.git
   cd Chat-With-Your-PDF
   ```
2. Install the required libraries:
   ```bash
    pip install -r requirements.txt
    ```
📚 **Requirements**:
- Python 3.12+
- OpenAI API Key (Get it from [OpenAI](https://platform.openai.com/))
- PDF files to query

📋 **Used Libraries:**
```bash
faiss-cpu==1.8.0.post1
langchain==0.3.1
langchain-community==0.3.1
langchain-core==0.3.6
langchain-openai==0.2.1
langchain-text-splitters==0.3.0
langgraph==0.2.28
langgraph-checkpoint==1.0.12
pypdf==5.0.1
streamlit==1.38.0
```
🚀 **Running the Project:**
1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Open the browser and go to `http://localhost:8501` to access the chatbot interface.
3. Upload the PDF file you want to query and start chatting with the chatbot.
4. Ask questions related to the content of the PDF file, and the chatbot will provide answers based on the information in the document.
5. Enjoy interacting with the RAG PDF Chatbot!

📝 **Note**: The chatbot is still in development, and improvements are being made to enhance its performance and capabilities. If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request, or contact me on linkedin.

👨‍💻 **Developed by**: [Mesut Duman](https://www.linkedin.com/in/mesut-duman/)

📄 **License**: This project is licensed under the Apache License 2.0.

🌟 **Screenshots**:
![main_page](https://github.com/user-attachments/assets/5f71b385-d3a9-42c2-a89e-92c28d594bdf)

```