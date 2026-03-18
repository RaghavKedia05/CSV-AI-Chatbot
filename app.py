import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.title("📊 CSV AI Chatbot")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    documents = df.astype(str).apply(
        lambda row: " | ".join(row), axis=1
    ).tolist()

    st.success("CSV Loaded Successfully!")

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    embeddings = model.encode(documents)

    # FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Query input
    query = st.text_input("Ask a question about your data:")

    if query:
        query_vector = model.encode([query])
        D, I = index.search(query_vector, k=3)

        results = [documents[i] for i in I[0]]

        st.subheader("Top Matches:")
        for r in results:
            st.write(r)
