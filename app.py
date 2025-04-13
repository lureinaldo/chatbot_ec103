import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuração da página
st.set_page_config(page_title="Chatbot RPPS")
st.title("🤖 Chatbot RPPS - Consulta à Nota Técnica SEI nº 12212/2019")

# Mensagem de carregamento
with st.spinner("Carregando e preparando a base de conhecimento..."):

    # Diretório de armazenamento vetorial
    persist_directory = "db"

    # Carrega o PDF
    loader = PyPDFLoader("biblioteca.pdf")  # Altere para o nome real do seu PDF se necessário
    documentos = loader.load()

    # Divide o texto em pedaços com sobreposição
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documentos)

    # Geração dos embeddings com HuggingFace
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Criação do banco vetorial
    vector_db = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=persist_directory)

    # Preparação do retriever
    retriever = vector_db.as_retriever(k=3)

    # Monta a cadeia RAG com prompt
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# Interface de entrada da pergunta
pergunta = st.text_input("Digite sua pergunta sobre a EC 103/2019 ou a Nota Técnica:")

# Exibe a resposta se houver pergunta
if pergunta:
    with st.spinner("Buscando resposta..."):
        resposta = rag_chain.invoke(pergunta)
        st.markdown("### 💬 Resposta:")
        st.write(resposta)
