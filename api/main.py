import os
import uuid
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from api.db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from api.chroma_utils import index_document_to_chroma, delete_doc_from_chroma
from api.langchain_utils import get_rag_chain  # lazy-load inside endpoint

# App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Health check (fast response)
@app.get("/")
def root():
    return {"status": "ok"}

# Lazy-load RAG chain inside endpoint
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session: {session_id}, Question: {query_input.question}, Model: {query_input.model.value}")

    # Get chat history
    chat_history = get_chat_history(session_id)

    # Lazy-load chain
    rag_chain = get_rag_chain(query_input.model.value)
    result = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })

    answer = result['answer']
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session: {session_id}, Answer: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

# Upload documents
@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed_extensions}")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(file.file.read())
        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_path, file_id)
        if success:
            return {"message": f"Uploaded and indexed {file.filename}", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# List docs
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

# Delete docs
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_ok = delete_doc_from_chroma(request.file_id)
    if chroma_ok:
        db_ok = delete_document_record(request.file_id)
        if db_ok:
            return {"message": f"Deleted document {request.file_id}"}
        else:
            return {"error": f"Deleted from Chroma but DB failed"}
    else:
        return {"error": f"Failed to delete document {request.file_id}"}

# Local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
