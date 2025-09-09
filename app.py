
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Carica variabili ambiente
load_dotenv()

# Caricamento CSS personalizzato
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Logo e intestazione
st.set_page_config(page_title="Chat del Corso", page_icon="https://www.istitutoleopardi.it/wp-content/uploads/Leo-favicon-1.gif")
st.markdown(
    '''
    <div class="logo-container" style="text-align: center;">
        <img src="https://www.istitutoleopardi.it/wp-content/uploads/LEO-logo.gif" width="139">
    </div>
    ''',
    unsafe_allow_html=True
)
st.title("💬 Chatbot dei POF dell'Istituto Leopardi")
st.markdown("❓ Fai una domanda basata sul materiale caricato nei POF dei vari licei.")

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
    if not docs:
        st.error("❌ Nessun contenuto valido trovato nei file PDF.")
        return None
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Errore nella creazione del vectorstore: {e}")
        return None

vectorstore = load_data()
if vectorstore is None:
    st.stop()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Campo input + svuotamento dopo invio
st.text_input(
    "Scrivi qui la tua domanda:",
    key="user_question",
    on_change=lambda: st.session_state.pop("asked", None)
)

if "user_question" in st.session_state and st.session_state.user_question and "asked" not in st.session_state:
    with st.spinner("Sto cercando nei materiali del corso..."):
        result = qa.run(st.session_state.user_question)
        st.success(result)
    st.session_state.asked = True
    st.session_state.user_question = ""
