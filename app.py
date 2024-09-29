import streamlit as st
from ingestion import PDFIngestor
from app_graph import PdfChat
st.set_page_config(page_title="Chat with your PDF", page_icon="🤖")
st.image("./media/cover.jpg", use_column_width=True)
if "messages" not in st.session_state:
    st.session_state.messages =  [{"role": "assistant", "content": "Time to talk about PDFs!"}]
    st.session_state.app = None
with st.sidebar:
    st.info("🤖 This app uses the OpenAI API to generate text, please provide your API key."
            "\n If you don't have an API key, you can get one [here](https://platform.openai.com/signup)."
            "\n You can also find the source code for this app [here](https://github.com/mesutdmn/Chat-With-Your-PDF)"
            "\n App keys are not stored or saved in any way.")
    openai_key = st.text_input("OpenAI API Key", type="password")

    if len(openai_key) < 1:
        st.error("Please enter your OpenAI API key")
        chat_active = False
    else:
        chat_active = True

    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

def initialize_ingestor(pdf_files):
    retriever = PDFIngestor(pdfs=pdf_files, api_key=openai_key).get_retriever()
    st.success("PDFs successfully uploaded")
    app = PdfChat(openai_key, retriever).graph
    st.success("ChatBot successfully initialized")
    return app

with st.sidebar:
    if st.button("Initialize ChatBot", type="primary"):
        st.session_state.app = initialize_ingestor(pdf_files)

app = st.session_state.app
def generate_response(question):
    return app.invoke(input={"question": question})["response"]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:= st.chat_input(placeholder="Ask a question", disabled= not chat_active):
    st.chat_message("user").markdown(question)

    st.session_state.messages.append({"role": "assistant", "content": question})

    response = generate_response(question)

    with st.chat_message("bot"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})




