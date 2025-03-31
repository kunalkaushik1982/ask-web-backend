from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial.distance import cosine
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import requests
import json
import os
from typing import List, Tuple, Dict, Any, Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Web Content Processing API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend URL instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Models
# ==========================================================

class QueryRequest(BaseModel):
    url: str
    question: str

class SummarizationRequest(BaseModel):
    url: str
    style: str  # "Concise", "Bullet Points", "Casual", "Professional"

class AnalyzeRequest(BaseModel):
    url: str

class RelatedLink(BaseModel):
    title: str
    url: str
    hostname: str

class RelatedLinksResponse(BaseModel):
    related_links: List[RelatedLink]

class SuggestionResponse(BaseModel):
    suggestions: Dict[str, Any]

class AnswerResponse(BaseModel):
    answer: str

class SummaryResponse(BaseModel):
    summary: str

# ==========================================================
# Global cache and constants
# ==========================================================

# Cache storage: {url: (chunks, chunk_embeddings)}
embedding_cache: Dict[str, Tuple[List[str], List[List[float]]]] = {}

SUMMARIZATION_STYLES = {
    "Concise": "Summarize the following text in a short and precise way.",
    "Bullet Points": "Summarize the following text using bullet points.",
    "Casual": "Summarize the following text in a friendly and conversational way.",
    "Professional": "Summarize the following text in a formal and professional manner."
}

# ==========================================================
# Utility Functions
# ==========================================================

def extract_api_key(authorization: str = Header(None)) -> str:
    """Extract and validate API key from authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    
    return authorization.split("Bearer ")[1].strip()

def fetch_page_content(url: str) -> str:
    """Fetch webpage content and extract only visible text using BeautifulSoup."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for script in soup(["script", "style", "meta", "noscript", "iframe"]):
            script.extract()

        # Extract text from relevant tags
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
        
        if not text_elements:
            return "No relevant text found on this page."
            
        # Join text with newlines to maintain paragraph structure
        text = "\n".join([p.get_text(separator=" ", strip=True) for p in text_elements])
        
        return text.strip()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching page: {str(e)}")

def get_or_create_embeddings(url: str, openai_api_key: str) -> Tuple[List[str], List[List[float]]]:
    """Get embeddings from cache or create and cache them if they don't exist."""
    if url in embedding_cache:
        return embedding_cache[url]
    
    # Fetch and split webpage content
    page_content = fetch_page_content(url)
    
    # Split content into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(page_content)

    # Compute embeddings and cache them
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chunk_embeddings = embedding_model.embed_documents(chunks)
    
    # Store in cache
    embedding_cache[url] = (chunks, chunk_embeddings)
    
    return chunks, chunk_embeddings

def get_most_relevant_chunks(query_embedding: List[float], 
                             chunk_embeddings: List[List[float]], 
                             chunks: List[str], 
                             top_k: int = 3) -> List[str]:
    """Find the top-k most relevant chunks using cosine similarity."""
    if not chunk_embeddings or not chunks:
        return []
        
    similarities = [1 - cosine(query_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]

# ==========================================================
# API Endpoints
# ==========================================================

@app.post("/process_page/", response_model=AnswerResponse)
async def process_page(request: QueryRequest, api_key: str = Depends(extract_api_key)):
    try:
        # Get or create embeddings for the URL
        chunks, chunk_embeddings = get_or_create_embeddings(request.url, api_key)
        
        # Create embedding model for the query
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        query_embedding = embedding_model.embed_query(request.question)
        
        # Get the most relevant chunks
        relevant_chunks = get_most_relevant_chunks(query_embedding, chunk_embeddings, chunks)
        
        # If no relevant chunks found, inform the user
        if not relevant_chunks:
            return {"answer": "I couldn't find relevant information to answer your question on this page."}
        
        # Generate response using LLM
        context = "\n".join(relevant_chunks)
        llm = ChatOpenAI(openai_api_key=api_key)
        response = llm.invoke(
            f"Answer the question based on this context:\n\n{context}\n\nQuestion: {request.question}"
        )
        
        # Extract answer text
        answer = response.content if hasattr(response, "content") else str(response)
        
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/summarize_page/", response_model=SummaryResponse)
async def summarize_page(request: SummarizationRequest, api_key: str = Depends(extract_api_key)):
    # Validate summarization style
    if request.style not in SUMMARIZATION_STYLES:
        valid_styles = ", ".join(SUMMARIZATION_STYLES.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid summarization style. Valid options are: {valid_styles}"
        )

    try:
        # Fetch page content
        page_content = fetch_page_content(request.url)
        
        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_text(page_content)
        
        # Limit to 5 chunks to prevent token overload
        chunks = chunks[:5]
        
        # Initialize LLM
        llm = ChatOpenAI(openai_api_key=api_key)
        
        # Process each chunk separately
        summaries = []
        for chunk in chunks:
            prompt = f"{SUMMARIZATION_STYLES[request.style]}\n\nText:\n{chunk}"
            response = llm.invoke(prompt)
            chunk_summary = response.content if hasattr(response, "content") else str(response)
            summaries.append(chunk_summary)
        
        # Combine summaries
        final_summary = "\n".join(summaries)
        
        return {"summary": final_summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing page: {str(e)}")

@app.post("/get_suggestions/", response_model=SuggestionResponse)
async def get_suggestions(request: AnalyzeRequest, api_key: str = Depends(extract_api_key)):
    try:
        # Initialize LLM
        llm = ChatOpenAI(openai_api_key=api_key)
        
        # Fetch page content
        page_content = fetch_page_content(request.url)
        
        # Define prompt for entity extraction
        prompt = """
        Task: Extract the four most important entities from the following text:
        "{text}"

        Instructions:
        1. Identify the four key entities (people, organizations, locations, dates, concepts, or other critical elements) that best represent the core meaning of the text.
        2. Ensure that the selected entities capture the essence of the content.
        3. Construct a **concise and meaningful phrase** (4 to 6 words) using these four entities.
        4. Make sure all the phrases should have meaning also avoid any special characters like "@,#,$,% etc"
        5. **Return the result in a structured JSON format.**

        ### **Example Output:**
        {{
            "entities": ["LangGraph", "LangChain", "LLM", "AI Capabilities"],
            "phrases": [
                "LangGraph vs LangChain differences",
                "LangGraph Platform deployment options",
                "LangGraph open source license",
                "LangGraph OSS LLM compatibility"
            ]
        }}
        """.format(text=page_content)
        
        # Get response from LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Parse JSON response
        try:
            suggestions = json.loads(response.content)
            return {"suggestions": suggestions}
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse suggestions from LLM")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting suggestions: {str(e)}")

@app.get("/fetch_related_links", response_model=RelatedLinksResponse)
async def fetch_related_links(query: str):
    # Get environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")
    google_search_api = os.getenv("GOOGLE_SEARCH_API")
    
    # Validate necessary environment variables
    if not all([google_api_key, search_engine_id, google_search_api]):
        raise HTTPException(
            status_code=500, 
            detail="Missing required environment variables for Google search"
        )
    
    # Set up search parameters
    params = {
        "key": google_api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 5  # Fetch top 5 results
    }

    try:
        # Send request to Google Custom Search API
        response = requests.get(google_search_api, params=params)
        response.raise_for_status()
        
        # Parse results
        search_results = response.json().get("items", [])
        
        # Format results
        related_links = []
        for item in search_results:
            link = item.get("link")
            title = item.get("title")
            hostname = requests.utils.urlparse(link).netloc
            
            related_links.append({
                "title": title,
                "url": link,
                "hostname": hostname
            })
        
        return {"related_links": related_links}
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching related links: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)