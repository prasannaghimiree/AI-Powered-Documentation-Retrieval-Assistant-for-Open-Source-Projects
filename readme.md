# **AI-Powered Documentation Assistant**

An AI-driven tool that simplifies understanding and contributing to open-source projects by analyzing repositories and documentation. This assistant provides automatic documentation summaries, answers user queries interactively, and generates dynamic contribution guides. It leverages **Ollama's Mistral LLM** and **all-MiniLM-v3 embeddings** with a **ChromaDB vector database** for efficient and scalable text retrieval.

---

## 🚀 **Features**

- **Automatic Summaries**: Retrieve information of project documentation.  
- **Interactive Q&A**: Ask questions about the repository and receive accurate answers.  
- **Dynamic Contribution Guides**: Automatically create "How to Contribute" guides for the repository.  
- **Efficient Search**: Search across project documentation and codebase using embeddings for relevant content.  

---

## 🛠️ **Tech Stack**

- **Language Model**: [Ollama Mistral LLM](https://github.com/ollama/ollama)  
- **Embeddings**: [Ollama all-MiniLM](https://github.com/ollama/ollama)  
- **Vector Database**: [ChromaDB](https://github.com/chroma-core/chroma)  
- **Backend**: Python, LangChain, FastApi 


---

## 📦 **Setup and Installation**

### Prerequisites
Ensure you have the following installed:
- **Python**
- **Download**[Ollama](https://ollama.com/download) 
- **Open Command prompt**: 
1. ollama pull mistral
2. ollama pull all-minilm



### Steps to Install

1. **Clone the Repository**
   ```bash
   git clone https://github.com/prasannaghimiree/AI-Powered-Documentation-Retrieval-Assistant-for-Open-Source-Projects.git
   cd langchain_project_naxa


