# LangChain RAG Chatbot (FastAPI + Streamlit) 

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that uses **LangChain**, **Google Gemini**, and **ChromaDB** to answer questions based on uploaded documents.

---

## Features 
- Upload multiple PDFs, DOCX, or HTML files  
- Retrieve context from your files (via ChromaDB)  
- Use Google Gemini via LangChain for contextual answers  
- Maintain session history and conversational context  
- FastAPI backend + Streamlit frontend  
- SQLite database for logs & documents  

---

## Tech Stack 
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **Database:** SQLite  
- **Vector Store:** ChromaDB  
- **LLM:** Google Gemini (via LangChain)  
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`

---

## Setup & Run 

git bash
git clone https://github.com/JashLadhani07/RAG-Chatbot.git
cd RAG-Chatbot
pip install -r requirements.txt

# Run backend
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run frontend
streamlit run app/app.py

---

## Envirourment Variables
GOOGLE_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
CHROMA_DB_PATH=./chroma_db
API_URL=http://localhost:8000

---

## Project Structure
RAG-Chatbot/
│
├── api/                # FastAPI backend
│   ├── main.py
│   ├── db_utils.py
│   ├── chroma_utils.py
│   ├── langchain_utils.py
│   └── pydantic_models.py
│
├── app/                # Streamlit frontend
│   ├── app.py
│   ├── chat_interface.py
│   └── sidebar.py
│
├── chroma_db/          # Vector database
├── rag_app.db          # SQLite database
├── requirements.txt
├── .gitignore
└── README.md
