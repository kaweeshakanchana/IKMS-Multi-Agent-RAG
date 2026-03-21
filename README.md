# 🧠 Multi-Agent Retrieval-Augmented Knowledge System

![AI](https://img.shields.io/badge/AI-Multi--Agent_Architecture-0052CC?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-6f42c1?style=flat-square) ![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-000000?style=flat-square) ![Next.js](https://img.shields.io/badge/Frontend-UI-000000?style=flat-square&logo=react)

A fully automated, intelligent system designed to answer complex questions from PDF documents with high accuracy and reduced hallucination, ensuring every response is strictly grounded in retrieved context.

Instead of a traditional single-step RAG flow, this application utilizes a **Multi-Agent Architecture** powered by **LangGraph**, where specialized AI agents collaborate behind the scenes to orchestrate complex reasoning and verification workflows.

## 🚀 Live Demo

**Try it out here:** [https://multi-agent-rag-system.vercel.app](https://lnkd.in/gEr7j45H)  
*(Note: Replace the display text with your actual deployed URL)*

## ✨ Features

* **Multi-Agent Orchestration:** Specialized AI agents handle distinct parts of the Retrieval-Augmented Generation process:
  * 🗺️ **Planning Agent:** Analyzes complex or ambiguous user questions, identifies key entities, and generates a structured, multi-step search strategy.
  * 🔍 **Retrieval Agent:** Executes semantic searches over the Pinecone vector index to gather the most relevant document chunks based on the plan.
  * 📝 **Summarization Agent:** Synthesizes a clean, accurate answer using *only* the retrieved context, preventing outside knowledge from creeping in.
  * 🛡️ **Verification Agent:** Acts as an anti-hallucination guardrail by auditing the final response against the original retrieved text before showing it to the user.

* **Semantic Document Understanding:** Leverages advanced vector embeddings to comprehend the actual meaning of your documents.
* **Production-Ready Backend:** Robust FastAPI endpoints for handling real-time QA and dynamic PDF indexing.

## 🛠️ Tech Stack

* **Framework Foundation:** LangChain v1.0
* **Agent Orchestration:** LangGraph
* **Vector Database:** Pinecone
* **Backend Framework:** FastAPI
* **LLM Provider:** OpenAI

## 💻 Getting Started

### Prerequisites
* Python 3.10+
* Pinecone API Key
* OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
