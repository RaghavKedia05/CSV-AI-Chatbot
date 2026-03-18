import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="CSV AI Chatbot",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# CUSTOM UI STYLING
# ---------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.title("📊 CSV AI Chatbot")
st.markdown("Upload your CSV and chat with your data using AI embeddings 🚀")

# ---------------------------
# LOAD MODEL (CACHED)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Convert rows → text
    documents = df.astype(str).apply(
        lambda row: " | ".join(row), axis=1
    ).tolist()

    # ---------------------------
    # CREATE EMBEDDINGS (CACHED)
    # ---------------------------
    @st.cache_data
    def create_embeddings(docs):
        return np.array(model.encode(docs))

    embeddings = create_embeddings(documents)

    # ---------------------------
    # FAISS INDEX
    # ---------------------------
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ---------------------------
    # CHAT INTERFACE
    # ---------------------------
    st.subheader("💬 Chat with your Data")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask something about your CSV...")

    if user_query:

        # Add user message
        st.session_state.chat_history.append(("user", user_query))

        # Search
        query_vector = model.encode([user_query])
        D, I = index.search(query_vector, k=3)

        results = [documents[i] for i in I[0]]

        response = "### 🔍 Top Relevant Data:\n"
        for r in results:
            response += f"- {r}\n"

        # Add bot response
        st.session_state.chat_history.append(("bot", response))

    # ---------------------------
    # DISPLAY CHAT
    # ---------------------------
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")
