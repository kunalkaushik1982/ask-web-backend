from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from scipy.spatial.distance import cosine
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import requests

import os
load_dotenv()
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize components
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI()

class QueryRequest(BaseModel):
    url: str
    question: str

def fetch_page_content(url):
    """Fetch webpage content using requests."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        #print(f"[INFO] Fetching content from URL: {url}")  # Debugging
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        #print("[INFO] Successfully fetched webpage content")  # Debugging
        return response.text
    except requests.exceptions.RequestException as e:
        #print(f"[ERROR] Failed to fetch page: {e}")  # Debugging
        raise HTTPException(status_code=400, detail=f"Error fetching page: {str(e)}")

def get_most_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Find the top-k most relevant chunks using cosine similarity."""
    #print("[INFO] Calculating cosine similarity between query and document chunks")  # Debugging
    similarities = [1 - cosine(query_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    #print(f"[INFO] Top {top_k} relevant chunks selected")  # Debugging
    return [chunks[i] for i in top_indices]

@app.post("/process_page/")
async def process_page(request: QueryRequest):
    try:
        #print(f"[INFO] Received request for URL: {request.url} with question: {request.question}")  # Debugging
        
        # Fetch webpage content
        page_content = fetch_page_content(request.url)
        #print(f"[INFO] Page content length: {len(page_content)} characters")  # Debugging

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(page_content)
        #print(f"[INFO] Split webpage into {len(chunks)} chunks")  # Debugging

        # Compute embeddings for chunks
        chunk_embeddings = embedding_model.embed_documents(chunks)
        #print(f"[INFO] Generated embeddings for {len(chunks)} chunks")  # Debugging

        # Compute query embedding
        query_embedding = embedding_model.embed_query(request.question)
        #print("[INFO] Generated query embedding")  # Debugging

        # Retrieve top relevant chunks
        relevant_chunks = get_most_relevant_chunks(query_embedding, chunk_embeddings, chunks)

        # Generate response
        context = "\n".join(relevant_chunks)
        #print(f"[INFO] Context selected for LLM:\n{context[:500]}...")  # Debugging (Prints first 500 chars)
        # response = llm.predict(f"Answer the question based on this context:\n\n{context}\n\nQuestion: {request.question}")
        # print (response)
        response = llm.invoke(f"Answer the question based on this context:\n\n{context}\n\nQuestion: {request.question}")
        #print (response)
        #print(f"[INFO] LLM response: {response}")  # Debugging

        # Extract answer from response
        if isinstance(response, dict) and "content" in response:
            response_text = response["content"]
        else:
            response_text = str(response.content)  # Fallback handling

        return {"answer": response_text}

    except Exception as e:
        #print(f"[ERROR] {str(e)}")  # Debugging
        raise HTTPException(status_code=500, detail=str(e))