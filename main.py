# from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from typing import List
# # from ollama_mistralllm import qa  

# app = FastAPI()


# # # Define the model for user queries
# # class Query(BaseModel):
# #     query: str
# #     chat_history: List[str] = []


# @app.get("/")
# def root():
#     return {"message": "Welcome to the FastAPI of Github Tasking Manager: Naxa"}


# # @app.post("/query")
# # def ask_query(query: Query):
# #     try:
# #         # Pass query and chat history to the QA model
# #         result = qa({'query': query.query, "chat_history": query.chat_history})
# #         answer = result.get('result', '').split("Answer:")[-1].strip()
# #         return {"query": query.query, "answer": answer, "chat_history": query.chat_history}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")





# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from typing import List
# # from langchain_ollama import OllamaLLM
# # from langchain.chains import RetrievalQA
# # from langchain.vectorstores import Chroma
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.prompts.prompt import PromptTemplate

# # app = FastAPI()

# # # Define the model for user queries
# # class Query(BaseModel):
# #     query: str
# #     chat_history: List[str] = []

# # # Initialize LangChain components
# # vector_saving_directory = "./chromaaa_db"

# # # Load vector store
# # vectorstore = Chroma(
# #     collection_name="github-documents",
# #     persist_directory=vector_saving_directory,
# #     embedding_function=None  # Embeddings should already be stored in the vector DB
# # )

# # # Set up the retriever
# # retriever = vectorstore.as_retriever(
# #     search_type="mmr",
# #     search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5}
# # )

# # # Define the LLM
# # llm = OllamaLLM(
# #     model="mistral",
# #     config={'max_new_tokens': 1000, 'temperature': 0.8}
# # )

# # # Define prompt template
# # prompt_template = """
# # Use the following piece of information to answer the user's question.
# # If you don't know the answer, just say that you don't know. Don't try to make up an answer.

# # Context: {context}
# # Question: {question}
# # Answer:
# # """
# # PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # # Initialize the QA chain
# # qa = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     chain_type="stuff",
# #     retriever=retriever,
# #     return_source_documents=False,
# #     chain_type_kwargs={"prompt": PROMPT}
# # )

# # # Define the FastAPI routes
# # @app.get("/")
# # def root():
# #     return {"message": "Welcome to the QA FastAPI!"}

# # @app.post("/query")
# # def ask_query(query: Query):
# #     try:
# #         # Pass query and chat history to the QA model
# #         result = qa({'query': query.query, "chat_history": query.chat_history})
# #         answer = result.get('result', '').split("Answer:")[-1].strip()
# #         return {"query": query.query, "answer": answer, "chat_history": query.chat_history}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama_mistralllm import fetch_data, split_text, initialize_vectorstore, initialize_retriever, initialize_qa_chain

app = FastAPI()

# Global variables for vectorstore and QA chain
vectorstore = None
qa_chain = None


class Question(BaseModel):
    query: str
    chat_history: list[str] = []


@app.on_event("startup")
def setup():
    """
    Setup event to initialize the QA system.
    """
    global vectorstore, qa_chain

    base_url = "https://raw.githubusercontent.com/hotosm/tasking-manager/develop/docs/developers/"
    files = [
        "code_of_conduct.md",
        "contributing-guidelines.md",
        "contributing.md",
        "development-setup.md",
        "error_code.md",
        "review-pr.md",
        "submit-pr.md",
        "tmschema.md",
        "translations.md",
        "versions-and-releases.md",
    ]

    try:
        raw_text = fetch_data(base_url, files)
        texts = split_text(raw_text)

        vectorstore = initialize_vectorstore(texts, embedding_model="all-minilm")
        retriever = initialize_retriever(vectorstore)
        qa_chain = initialize_qa_chain(retriever)
    except Exception as e:
        print(f"Failed to initialize QA system: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize QA system")


@app.post("/ask")
def ask_question(question: Question):
    """
    Endpoint to ask a question and get an answer.
    """
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="QA system is not initialized.")

    try:
        result = qa_chain({'query': question.query, 'chat_history': question.chat_history})
        answer = result.get('result', '').strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
