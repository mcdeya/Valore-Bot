"""
RAG-Based Valorant Lore Chatbot — Streamlit App
Pipeline: Document Loading -> Chunking -> all-MiniLM-L6-v2 Embeddings -> ChromaDB -> Groq (OpenAI GPT-OSS-20B)

Run locally with:
    pip install streamlit langchain-classic langchain-community langchain-groq langchain-huggingface chromadb pypdf unstructured sentence-transformers
    streamlit run streamlit_app.py
"""

import os

import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="VALORANT // LORE ARCHIVE", page_icon="🎯", layout="wide")

# ---------------------------------------------------------------------------
# Design system — tactical HUD theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

    :root {
        --void: #0A0E13;
        --panel: #121A22;
        --panel-2: #0E151C;
        --line: #263341;
        --red: #FF4655;
        --red-dim: #7A2530;
        --ivory: #ECE8E1;
        --steel: #7D8B94;
        --cyan: #21E6C1;
        --gold: #FFB94A;
    }

    html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }
    .stApp { background: var(--void); color: var(--ivory); }

    h1, h2, h3, h4, .hero-title { font-family: 'Rajdhani', sans-serif; letter-spacing: 0.02em; }

    /* Hero header */
    .hero {
        position: relative;
        background: linear-gradient(120deg, var(--panel) 0%, var(--panel-2) 100%);
        border: 1px solid var(--line);
        clip-path: polygon(0 0, 100% 0, 100% 85%, 97% 100%, 0 100%);
        padding: 28px 36px 22px 36px;
        margin-bottom: 20px;
    }
    .hero::before {
        content: "";
        position: absolute; left: 0; top: 0; bottom: 0;
        width: 5px;
        background: var(--red);
    }
    .hero-eyebrow {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: var(--red);
        margin: 0 0 6px 0;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--ivory);
        margin: 0;
        line-height: 1.05;
    }
    .hero-sub {
        font-family: 'Barlow', sans-serif;
        color: var(--steel);
        font-size: 0.98rem;
        margin-top: 10px;
        max-width: 640px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--panel-2);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ivory);
        font-size: 0.95rem;
        border-bottom: 1px solid var(--line);
        padding-bottom: 8px;
        margin-top: 18px;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stMarkdown {
        color: var(--steel);
    }

    /* Buttons */
    .stButton > button, [data-testid="stBaseButton-primary"] {
        background: var(--red) !important;
        color: var(--void) !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        clip-path: polygon(6% 0, 100% 0, 94% 100%, 0 100%);
        transition: filter 0.15s ease;
    }
    .stButton > button:hover { filter: brightness(1.15); }

    /* Text inputs, selects, number inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
        background: var(--panel) !important;
        color: var(--ivory) !important;
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
    }

    /* Chat messages as intel panels */
    [data-testid="stChatMessage"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 3px solid var(--steel);
        border-radius: 0;
        padding: 4px 6px;
        margin-bottom: 10px;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-left: 3px solid var(--red);
    }

    /* Chat input bar */
    [data-testid="stChatInput"] {
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        background: var(--panel) !important;
    }

    /* Expanders (retrieved source chunks) */
    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 0 !important;
        background: var(--panel-2) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.85rem;
        color: var(--steel) !important;
    }

    hr, [data-testid="stDivider"] { border-color: var(--line) !important; }

    /* Alerts */
    [data-testid="stAlert"] { border-radius: 0 !important; border-left-width: 3px !important; }

    /* API key status card */
    .key-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 4px 0 16px 0;
    }
    .key-card-label {
        font-family: 'Barlow', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--ivory);
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    .key-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Barlow', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 5px 12px;
        border-radius: 999px;
    }
    .key-badge.ok {
        background: rgba(31, 237, 176, 0.12);
        border: 1px solid #1FEDB0;
        color: #1FEDB0;
    }
    .key-badge.missing {
        background: rgba(255, 70, 85, 0.12);
        border: 1px solid var(--red);
        color: var(--red);
    }

    /* Suggestion chips */
    .chip-eyebrow {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--steel);
        margin: 4px 0 10px 2px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div.chip-marker) {
        background: var(--panel) !important;
        border-radius: 0 !important;
        transition: filter 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div > div.chip-marker):hover {
        filter: brightness(1.08);
    }
    .chip-cat {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin: 2px 0 4px 0;
    }
    .chip-cat.locations { color: var(--cyan); }
    .chip-cat.agent { color: var(--gold); }
    .chip-cat.timeline { color: var(--red); }
    .chip-q {
        font-family: 'Barlow', sans-serif;
        font-weight: 500;
        font-size: 0.92rem;
        color: var(--ivory);
        line-height: 1.3;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
        background: transparent !important;
        color: transparent !important;
        border: none !important;
        clip-path: none !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        position: absolute;
        inset: 0;
        width: 100%;
    }
    </style>

    <div class="hero">
        <p class="hero-eyebrow">LORE DATABASE</p>
        <p class="hero-title">VALORANT LORE ARCHIVE</p>
        <p class="hero-sub">
            Query the agent, location, and timeline records below. Every answer is grounded in the
            documents indexed in the sidebar — nothing is drawn from outside the archive.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("OPERATOR CONSOLE")

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    key_status_class = "ok" if groq_api_key else "missing"
    key_status_text = "✓ API key loaded" if groq_api_key else "✕ API key missing"
    st.markdown(
        f"""
        <div class="key-card">
            <div class="key-card-label">🔑 Groq API key</div>
            <div class="key-badge {key_status_class}">{key_status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not groq_api_key:
        st.caption("Add it under Advanced settings → Secrets in Streamlit Cloud, or set it as an environment variable locally.")

    st.subheader("Parameters")
    top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=25, value=25)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    fallback_enabled = st.checkbox("Enable fallback instruction", value=True)

    # Fixed defaults (not shown in the UI)
    chunk_size = 500
    chunk_overlap = 50
    model_name = "openai/gpt-oss-20b"

    st.divider()

    repo_data_dir = "my_data"

    if st.session_state.doc_count:
        st.success(f"{st.session_state.doc_count} document(s) indexed → {st.session_state.chunk_count} chunks")

# ---------------------------------------------------------------------------
# Build the RAG pipeline
# ---------------------------------------------------------------------------
def build_rag_chain(repo_data_dir, chunk_size, chunk_overlap, top_k,
                     model_name, temperature, fallback_enabled, api_key):
    if not api_key:
        st.error("Groq API key not found. Add it under Advanced settings → Secrets in Streamlit Cloud.")
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

    if not os.path.isdir(repo_data_dir):
        st.error(f"`{repo_data_dir}/` folder not found. Commit your documents into it on GitHub.")
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


current_params = (top_k, temperature, fallback_enabled)
needs_build = (
    st.session_state.rag_chain is None
    or st.session_state.get("last_params") != current_params
)

if needs_build and groq_api_key:
    with st.spinner("Loading documents, chunking, embedding, and indexing..."):
        st.session_state.rag_chain = build_rag_chain(
            repo_data_dir, chunk_size, chunk_overlap, top_k,
            model_name, temperature, fallback_enabled, groq_api_key,
        )
        st.session_state.last_params = current_params
        st.session_state.messages = []

# ---------------------------------------------------------------------------
# Suggestion chips (shown before the first message)
# ---------------------------------------------------------------------------
SUGGESTED_QUESTIONS = [
    ("Locations", "Where is Ascent?"),
    ("Agent", "What country is Fade from?"),
    ("Timeline", "What is Radianite?"),
    ("Locations", "What company is connected to Pearl?"),
    ("Agent", "Who is Raze?"),
    ("Timeline", "When did the Alpha-Omega conflict begin?"),
]

if not st.session_state.messages:
    st.markdown('<p class="chip-eyebrow">Suggested queries</p>', unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for i, (category, question) in enumerate(SUGGESTED_QUESTIONS):
        with chip_cols[i % 3]:
            with st.container(border=True):
                st.markdown('<div class="chip-marker"></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<p class="chip-cat {category.lower()}">{category}</p>'
                    f'<p class="chip-q">{question}</p>',
                    unsafe_allow_html=True,
                )
                if st.button(question, key=f"chip_{i}"):
                    st.session_state.pending_query = question
                    st.rerun()
    st.divider()

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Retrieved intel fragments"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Fragment {i}** — source: `{src}`")

user_query = st.chat_input("Ask about agents, locations, or the timeline...")

if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    if st.session_state.rag_chain is None:
        st.warning("Archive not indexed yet — check that GROQ_API_KEY is set and my_data/ contains documents.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Scanning archive..."):
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                sources = [doc.metadata.get("source", "Unknown") for doc in response.get("context", [])]
            st.markdown(answer)
            if sources:
                with st.expander("Retrieved intel fragments"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Fragment {i}** — source: `{src}`")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

# ---------------------------------------------------------------------------
# Footer / reset
# ---------------------------------------------------------------------------
st.divider()
col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Reset Console"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.caption(
        "Tip: to reproduce the hallucination stress test, turn OFF 'Enable fallback instruction', "
        "raise Temperature to 1.0 (the archive re-indexes automatically), then ask an "
        "out-of-domain question (e.g. 'Who is Manny Pacquiao?')."
    )
