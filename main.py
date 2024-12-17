from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ollama_mistralllm import qa  

app = FastAPI()

class Query(BaseModel):
    query: str
    chat_history: List[str] = []


@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI of Github Tasking Manager: Naxa"}


@app.post("/query")
def ask_query(query: Query):
    try:
        # Pass query and chat history to the QA model
        result = qa({'query': query.query, "chat_history": query.chat_history})
        answer = result.get('result', '').split("Answer:")[-1].strip()
        return {"query": query.query, "answer": answer, "chat_history": query.chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


