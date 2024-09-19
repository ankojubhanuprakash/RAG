# RAG (Retrieval-Augmented Generation) Proof of Concept

This project demonstrates a simple implementation of **Retrieval-Augmented Generation (RAG)** using **Universal Sentence Encoder (USE)** for document embeddings and **Microsoft's PHI** model for generating responses. The project is containerized using Docker for easy deployment.

## How It Works

- The system retrieves relevant documents from a corpus based on a query using **Universal Sentence Encoder (USE)** and **FAISS**.
- The retrieved documents are combined with the query and passed to **Microsoft's PHI** model for generating a coherent response.
- A simple Flask API is provided to interact with the RAG system.

## Technologies Used

- Universal Sentence Encoder (USE) for embeddings
- FAISS for document retrieval
- Microsoft's PHI model for generation
- Flask for creating an API
- Docker for containerization

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-poc.git
cd rag-poc
