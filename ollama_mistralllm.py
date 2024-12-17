import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA


def fetch_data(base_url, files):

    raw_text = ""
    for file_name in files:
        url = base_url + file_name
        try:
            response = requests.get(url)
            response.raise_for_status()
            raw_text += response.text + "\n\n"
            print(f"Fetched {file_name}")
        except Exception as e:
            print(f"Failed to fetch {file_name}: {e}")
    print("All files fetched successfully.")
    return raw_text
    


def split_text(raw_text, chunk_size=800, chunk_overlap=200):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def initialize_retriever(vectorstore):
   
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5}
    )

def initialize_vectorstore(texts, embedding_model, directory="./chromaaa_db"):

    embeddings = OllamaEmbeddings(model=embedding_model)

    if isinstance(texts[0], str):
        texts_content = texts
        metadatas = [{"source": "unknown"} for _ in texts_content]
    else:
        texts_content = [doc.page_content for doc in texts]
        metadatas = [{"source": doc.metadata["source"]} for doc in texts]

    vectorstore = Chroma(
        collection_name="github-documents",
        embedding_function=embeddings,
        persist_directory=directory
    )

    vectorstore.add_texts(texts=texts_content, metadatas=metadatas)
    return vectorstore
    


def initialize_qa_chain(retriever, model="mistral", max_tokens=1000, temperature=0.8):

    llm = OllamaLLM(
        model=model,
        config={'max_new_tokens': max_tokens, 'temperature': temperature}
    )

    prompt_template = """
    Use the following piece of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    And do not repeat the same answer. Give the answer if the query is in the dataset. Do not give random answers.

    Context: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

def ask_question(qa_chain, query, chat_history=None):
 
    if chat_history is None:
        chat_history = []
    try:
        result = qa_chain({'query': query, "chat_history": chat_history})
        answer = result.get('result', '').split("Answer:")[-1].strip()
        return answer
    except Exception as e:
        return f"An error occurred: {e}"



def main():
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

    
    raw_text = fetch_data(base_url, files)
    texts = split_text(raw_text)
    vectorstore = initialize_vectorstore(texts, embedding_model="all-minilm")
    retriever = initialize_retriever(vectorstore)
    qa_chain = initialize_qa_chain(retriever)

  
    chat_history = []
    while True:
        query = input("Enter your question ")
        if query.lower() == 'exit':
            break
        answer = ask_question(qa_chain, query, chat_history)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()





