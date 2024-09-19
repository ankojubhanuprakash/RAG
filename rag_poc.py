import tensorflow_hub as hub
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the Universal Sentence Encoder model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load your corpus
with open("corpus.txt", "r") as f:
    documents = f.readlines()

# Step 1: Embed the documents using USE
doc_embeddings = embed(documents).numpy()

# Step 2: Create a FAISS index and add document embeddings
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Step 3: Define the retrieval function
def retrieve_docs(query, k=2):
    query_embedding = embed([query]).numpy()
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

# Load Microsoft's PHI model for generation
model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 4: Combine Retrieval and Generation
def generate_response(query):
    retrieved_docs = retrieve_docs(query)
    input_text = " ".join(retrieved_docs) + " " + query
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example Query
query = "Why do humans need water?"
response = generate_response(query)
print(f"Query: {query}\nResponse: {response}")
