
import streamlit as st
from dotenv import load_dotenv
import os
api_key = os.getenv("OPENAI_API_KEY")  # oppure use st.secrets["OPENAI_API_KEY"]
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Carica variabili ambiente
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="Chat del Corso", page_icon="📘")
st.title("💬 Chatbot del Corso - Istituto Leopardi")
st.markdown("❓ Fai una domanda basata sul materiale caricato nei POF.")

# Mostra il logo in cima alla pagina
st.image("logo.gif", width=200)

# Carica e processa più PDF
@st.cache_resource
def load_data():
    pdf_files = [
        "POFisttecnicoeconomico2024_25.pdf",
        "POFliceoeuropeo2024_25.pdf",
        "POFliceolinguistico2024_25.pdf",
        "POFliceoscienze_umane2024_25.pdf",
        "POFliceoscientifico2024_25.pdf"
    ]
    pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages.extend(loader.load_and_split())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Inizializza sistema RAG
vectorstore = load_data()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Campo di input
user_question = st.text_input("Scrivi qui la tua domanda:")

if user_question:
    with st.spinner("Sto cercando nei materiali del corso..."):
        result = qa.run(user_question)
        st.success(result)
