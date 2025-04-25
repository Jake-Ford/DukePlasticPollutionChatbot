import streamlit as st
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from transformers import AutoModel, AutoTokenizer

from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
from google import generativeai as genai
#import google.generativeai as genai
import requests

# --- Load environment variables ---
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")

@st.cache_data(show_spinner="Loading document embeddings...")
def load_embeddings():
    url = "https://drive.google.com/uc?export=download&id=1o988OjkTRNPY_B-zTqRQyT86dzklcdCP"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    return pickle.loads(b"".join(r.iter_content(chunk_size=8192)))

vectors, metadata = load_embeddings()


# --- Load embedding model ---
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embed_model():
    model_name = "intfloat/e5-base-v2"

    # Load tokenizer and model on CPU to avoid meta tensor issues
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = Transformer(model_name, max_seq_length=512)
    pooling = Pooling(word_embedding_dimension=transformer.get_word_embedding_dimension())
    
    return SentenceTransformer(modules=[transformer, pooling])

embed_model = get_embed_model()

# --- LLM API setup ---
def get_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://api.mistral.ai/v1/chat/completions", json=body, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

genai.configure(api_key=GEMMA_API_KEY)

def get_gemma_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit UI ---
st.title("🌍 Plastic Policy Chatbot")

query = st.text_input("Ask a question about global plastic regulations:")
llm_choice = st.selectbox("Choose an LLM", ["Mistral", "Gemma"])

if query:
    with st.spinner("Thinking..."):
        query_embedding = embed_model.encode(query).reshape(1, -1)
        embedding_matrix = np.vstack(vectors)
        sims = cosine_similarity(query_embedding, embedding_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:5]

        top_chunks = [metadata[i]["chunk"] for i in top_indices]
        top_sources = [metadata[i]["source"] for i in top_indices]

        context = "\n\n---\n\n".join(top_chunks)

        prompt = f"""
You are a policy expert. Answer the following question using the provided plastic policy documents.

Context:
{context}

Question:
{query}

Answer:
        """

        if llm_choice == "Gemma":
            answer = get_gemma_response(prompt)
        else:
            answer = get_mistral_response(prompt)

        st.markdown("### 💬 Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("### 📄 Top Sources")
        for src in set(top_sources):
            st.write(f"- {src}")
