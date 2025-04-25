# 🧠 Duke Plastic Policy Chatbot

Welcome to the **Duke Plastic Pollution Working Group**'s interactive chatbot project! This tool enables users to ask questions about global plastic pollution regulations using a powerful combination of document embeddings and large language models (LLMs).

🌐 **Live App**: [Try it on Streamlit Cloud →](https://duke-plastic-pollution-chatbot.streamlit.app/)

---

## 🚀 Overview

This chatbot is designed to:

- Search a corpus of global plastic policy documents
- Retrieve the most relevant document excerpts using vector similarity
- Generate accurate, policy-aware answers with LLMs (Gemma or Mistral)

---

## 🔍 Features

- 🧾 **Document RAG (Retrieval-Augmented Generation)** using `intfloat/e5-base-v2` embeddings
- 🧠 **LLM Integration**: Supports both [Gemma](https://ai.google.dev/gemma) and [Mistral](https://mistral.ai/)
- ⚡️ Fast similarity search with cosine distance
- 📄 Displays top source documents used for answering
- 🧼 Minimal UI using Streamlit

---

## 🛠️ Setup

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/dukeplasticpollutionchatbot.git
cd dukeplasticpollutionchatbot
