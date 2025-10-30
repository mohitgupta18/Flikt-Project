"""
chatbot_local.py
Same retrieval, but uses a small local generator (Flan-T5).
Note: Some machines may be slow or run out of RAM. Use small models like 'google/flan-t5-small'.
"""
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

VECTORSTORE = "../models/vectorstore/vector_store.npz"
META_CSV = "../models/vectorstore/meta.csv"
EMB_MODEL = "all-mpnet-base-v2"
GEN_MODEL = "google/flan-t5-small"  # change to flan-t5-base if you have RAM
TOP_K = 5

@st.cache_resource
def load_vectorstore(npz_path, meta_path):
    z = np.load(npz_path)
    return z["embeddings"], z["ids"].astype(str), pd.read_csv(meta_path)

@st.cache_resource
def load_embedder(name=EMB_MODEL):
    return SentenceTransformer(name)

@st.cache_resource
def load_generator(model_name=GEN_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def retrieve(query, embeddings, texts, embedder, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    sims = cosine_similarity(q_emb, embeddings)[0]
    idx = np.argsort(-sims)[:top_k]
    return idx, sims[idx]

def build_prompt(query, retrieved_texts):
    header = "Use ONLY the following feedback context to answer the user's question and suggest 3 concrete actions to improve customer satisfaction.\n\n"
    ctx = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(retrieved_texts)])
    return f"{header}{ctx}\n\nUser question: {query}\n\nAnswer and suggested actions:"

st.title("Feedback Chatbot (Local generator)")
embeddings, ids, meta = load_vectorstore(VECTORSTORE, META_CSV)
embedder = load_embedder()
tokenizer, model, device = load_generator()

query = st.text_input("Ask a question about feedback")
if st.button("Ask") and query.strip():
    idxs, scores = retrieve(query, embeddings, meta["summary"].values, embedder)
    retrieved = meta.iloc[idxs]["summary"].tolist()
    st.subheader("Retrieved context")
    for i, (r,s) in enumerate(zip(retrieved, scores), 1):
        st.markdown(f"**[{i}] (score: {s:.3f})** {r}")

    prompt = build_prompt(query, retrieved)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, num_beams=3)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.subheader("Answer + Actions")
    st.write(answer)
