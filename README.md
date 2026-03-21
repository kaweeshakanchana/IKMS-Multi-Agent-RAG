🤖 Multi-Agent Retrieval-Augmented Knowledge System📌 Project OverviewThis project implements an advanced Multi-Agent RAG (Retrieval-Augmented Generation) pipeline designed to answer questions from PDF documents with high accuracy and reduced hallucination. Moving beyond a standard single-step RAG flow, this system utilizes LangGraph to coordinate specialized AI agents that collaboratively plan, retrieve, summarize, and verify information.This repository specifically includes the implementation of an intelligent Query Planning & Decomposition Agent, which analyzes complex, multi-part questions and creates a structured search strategy before retrieval begins.✨ Key FeaturesSemantic Document Understanding: Leverages vector embeddings to comprehend the semantic meaning of user queries and documents.Intelligent Query Planning: Dynamically rephrases ambiguous questions and decomposes complex queries into focused sub-questions for better retrieval.Fast & Scalable Retrieval: Uses Pinecone as the vector database for efficient document storage and similarity search.Hallucination Prevention: Includes a dedicated verification layer to ensure all answers are strictly grounded in the retrieved context.Production-Ready Backend: Exposes the pipeline via a robust FastAPI backend with endpoints for PDF indexing and QA.🧠 Multi-Agent WorkflowThe system orchestrates the following linear flow: START → planning → retrieval → summarization → verification → END.Planning Agent: Analyzes the user's raw question, identifies key entities, and outputs a structured natural language search plan and decomposed sub-questions.Retrieval Agent: Executes semantic searches against the Pinecone index using both the original question and the generated plan.Summarization Agent: Synthesizes a clean, accurate answer using only the retrieved context chunks.Verification Agent: Audits the response to correct inaccuracies and eliminate hallucinations before returning the final answer to the user.🛠️ Technology StackCore Framework: LangChain v1.0 Agent Orchestration: LangGraph Vector Database: Pinecone Backend API: FastAPI LLM Provider: OpenAI📂 Key Project StructurePlaintext├── src/
│   ├── app/
│   │   ├── api.py                    # FastAPI endpoint definitions (/qa and /index-pdf) [cite: 38]
│   │   ├── core/
│   │   │   ├── agents/
│   │   │   │   ├── agents.py         # Retrieval, Summarization, Verification, and Planning agent nodes [cite: 23, 24, 25]
│   │   │   │   ├── graph.py          # LangGraph StateGraph wiring the linear QA flow [cite: 26, 27, 28]
│   │   │   │   ├── state.py          # QAState schema definition [cite: 29, 30]
│   │   │   │   └── tools.py          # Retrieval tools for Pinecone queries [cite: 32, 33]
│   │   │   └── retrieval/
│   │   │       ├── vector_store.py   # Pinecone setup and PDF indexing [cite: 34]
│   │   │       └── serialization.py  # Chunk-to-context-string conversion [cite: 35]
│   │   └── services/
│   │       └── qa_service.py         # Service facade over LangGraph QA flow [cite: 36, 37]
🚀 Getting StartedPrerequisitesPython 3.10+Pinecone Account & API KeyOpenAI API Key1. Clone the repositoryBashgit clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Set up the environmentCreate a virtual environment and install dependencies:Bashpython -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
3. Configure Environment VariablesCreate a .env file in the root directory and add your credentials:Code snippetOPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
PINECONE_INDEX_NAME=your_index_name
4. Run the ApplicationStart the FastAPI server:Bashuvicorn src.app.api:app --reload
The API will be available at http://localhost:8000. You can access the interactive Swagger documentation at http://localhost:8000/docs.(Note: Frontend UI setup instructions can be added here based on your specific implementation).🔗 LinksLive Demo: [Insert Link Here]LinkedIn Post: [Insert Link Here]
