import os.path

from typing import List
from langchain_community.llms import GPT4All
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain



# Constants
current_directory = os.path.dirname(__file__)
local_path = current_directory + "/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
model_path = current_directory + "/models/gpt4all-falcon-newbpe-q4_0.gguf"
pdf_path = current_directory + "/docs/sample.pdf"
index_path = current_directory + "/pdf_index"

# Functions
def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)

def load_documents() -> List:
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_chunks(sources: List) -> List:
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks

def generate_index(chunks: List, embeddings: LlamaCppEmbeddings) -> FAISS:
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Main execution
llm = GPT4All(model=local_path, max_tokens=2048, verbose=True)

embeddings = initialize_embeddings()
# sources = load_documents()
# chunks = split_chunks(sources)
# vectorstore = generate_index(chunks, embeddings)
# vectorstore.save_local("pdf_index")

index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

qa = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), max_tokens_limit=400)

# Chatbot loop
chat_history = []
print("Welcome to the chatbot! Type 'exit' to stop.")
while True:
    query = input("Please enter your question: ")

    if query.lower() == 'exit':
        break
    result = qa({"question": query, "chat_history": chat_history})

    print("Answer:", result['answer'])