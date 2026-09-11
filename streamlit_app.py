"""
RAG-Based Valorant Lore Chatbot — Streamlit App
Pipeline: Document Loading -> Chunking -> all-MiniLM-L6-v2 Embeddings -> ChromaDB -> Groq (OpenAI GPT-OSS-20B)

Run locally with:
    pip install streamlit langchain langchain-community langchain-groq langchain-huggingface chromadb pypdf unstructured
    streamlit run streamlit_app.py
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Valorant Lore RAG Chatbot", page_icon="🎯", layout="wide")
st.title("🎯 Valorant Lore RAG Chatbot")
st.caption("Retrieval-Augmented Generation over uploaded Valorant lore/timeline documents, powered by Groq (OpenAI GPT-OSS-20B).")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Your key is used only for this session and is not stored.",
    )

    st.subheader("Retrieval settings")
    chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=500, value=50, step=10)
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=25, value=25)

    st.subheader("Model settings")
    model_name = st.selectbox(
        "Groq model",
        options=["openai/gpt-oss-20b", "llama-3.1-8b-instant"],
        index=0,
    )
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    fallback_enabled = st.toggle(
        "Enable fallback instruction",
        value=True,
        help="When on, the model is told to reply with a fixed refusal if the answer isn't in the retrieved context. "
             "Turn this off (with a higher temperature) to reproduce the hallucination stress test.",
    )

    st.divider()
    st.subheader("📁 Domain documents")

    data_source = st.radio(
        "Document source",
        options=["From repo (my_data/ folder)", "Upload files"],
        index=0,
        help="'From repo' reads whatever .pdf/.txt files are committed to this app's my_data/ folder on GitHub. "
             "Use this for a deployed app with a fixed document set.",
    )

    uploaded_files = None
    repo_data_dir = "my_data"

    if data_source == "Upload files":
        uploaded_files = st.file_uploader(
            "Upload .pdf or .txt lore/timeline documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
    else:
        if os.path.isdir(repo_data_dir):
            repo_files = [f for f in os.listdir(repo_data_dir) if f.lower().endswith((".pdf", ".txt"))]
            if repo_files:
                st.caption(f"Found {len(repo_files)} file(s) in `{repo_data_dir}/`:")
                for f in repo_files:
                    st.caption(f"• {f}")
            else:
                st.warning(f"No .pdf/.txt files found in `{repo_data_dir}/`.")
        else:
            st.warning(
                f"No `{repo_data_dir}/` folder found next to this app. "
                "Create one in your GitHub repo and commit your .pdf/.txt files into it, "
                "or switch to 'Upload files' above."
            )

    build_clicked = st.button("Build / Rebuild Knowledge Base", type="primary", use_container_width=True)

    if st.session_state.doc_count:
        st.success(f"Loaded {st.session_state.doc_count} document(s) → {st.session_state.chunk_count} chunks")

# ---------------------------------------------------------------------------
# Build the RAG pipeline
# ---------------------------------------------------------------------------
def build_rag_chain(data_source, files, repo_data_dir, chunk_size, chunk_overlap, top_k,
                     model_name, temperature, fallback_enabled, api_key):
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        return None

    os.environ["GROQ_API_KEY"] = api_key

    def _load_and_index(source_dir):
        loader = DirectoryLoader(source_dir, glob="**/*.*", show_progress=False)
        raw_documents = loader.load()

        if not raw_documents:
            st.error(f"No documents could be loaded from `{source_dir}`.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents = text_splitter.split_documents(raw_documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        st.session_state.doc_count = len(raw_documents)
        st.session_state.chunk_count = len(documents)
        return retriever

    if data_source == "Upload files":
        if not files:
            st.error("Please upload at least one .pdf or .txt document.")
            return None
        with tempfile.TemporaryDirectory() as tmp_dir:
            for f in files:
                with open(os.path.join(tmp_dir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            retriever = _load_and_index(tmp_dir)
    else:
        if not os.path.isdir(repo_data_dir):
            st.error(f"`{repo_data_dir}/` folder not found. Commit your documents into it on GitHub, or switch to 'Upload files'.")
            return None
        retriever = _load_and_index(repo_data_dir)

    if retriever is None:
        return None

    llm = ChatGroq(model_name=model_name, temperature=temperature)

    if fallback_enabled:
        system_prompt = (
            "You are a specialized AI assistant for the user's uploaded domain.\n"
            "Answer questions strictly using ONLY the provided context below.\n"
            "If the answer cannot be found in the context, reply: "
            "'I cannot answer based on the provided domain data.'\n\n"
            "Context:\n{context}"
        )
    else:
        system_prompt = (
            "You are a specialized AI assistant for the user's uploaded domain.\n"
            "Answer questions using the provided context below.\n\n"
            "Context:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain


if build_clicked:
    with st.spinner("Loading documents, chunking, embedding, and indexing..."):
        st.session_state.rag_chain = build_rag_chain(
            data_source, uploaded_files, repo_data_dir, chunk_size, chunk_overlap, top_k,
            model_name, temperature, fallback_enabled, groq_api_key,
        )
        st.session_state.messages = []
    if st.session_state.rag_chain:
        st.sidebar.success("Knowledge base built successfully.")

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Retrieved source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}** — source: `{src}`")

user_query = st.chat_input("Ask a question about the uploaded domain...")

if user_query:
    if st.session_state.rag_chain is None:
        st.warning("Build the knowledge base first using the sidebar (upload documents, then click 'Build / Rebuild Knowledge Base').")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                sources = [doc.metadata.get("source", "Unknown") for doc in response.get("context", [])]
            st.markdown(answer)
            if sources:
                with st.expander("Retrieved source chunks"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}** — source: `{src}`")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

# ---------------------------------------------------------------------------
# Footer / reset
# ---------------------------------------------------------------------------
st.divider()
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.caption(
        "Tip: to reproduce the hallucination stress test, turn OFF 'Enable fallback instruction', "
        "raise Temperature to 1.0, click 'Build / Rebuild Knowledge Base' again, then ask an "
        "out-of-domain question (e.g. 'Who is Manny Pacquiao?')."
    )
