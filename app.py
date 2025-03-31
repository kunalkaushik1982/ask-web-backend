from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import numpy as np
from scipy.spatial.distance import cosine
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
import requests
import json

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    url: str
    question: str

class SummarizationRequest(BaseModel):
    url: str
    style: str  # Can be "Concise", "Bullet Points", "Casual", "Professional"

class AnalyzeRequest(BaseModel):
    url:str

# Cache storage: {url: (chunks, chunk_embeddings)}
embedding_cache = {}

def fetch_page_content_old(url):
    """Fetch webpage content using requests."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching page: {str(e)}")

def fetch_page_content(url):
    """Fetch webpage content and extract only visible text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements (JavaScript, CSS)
        for script in soup(["script", "style", "meta", "noscript", "iframe"]):
            script.extract()  # Removes the tag from the soup

        # Extract text and maintain paragraph order
        text = "\n".join([p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])])

        # Ensure text is not empty
        if not text.strip():
            return "No relevant text found on this page."

        return text

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching page: {str(e)}")


def get_most_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    """Find the top-k most relevant chunks using cosine similarity."""
    similarities = [1 - cosine(query_embedding, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

@app.post("/process_page/")
async def process_page(request: QueryRequest, authorization: str = Header(None)):
    # Extract API key from Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Missing or invalid API key")

    openai_api_key = authorization.split("Bearer ")[1].strip()

    try:
        
        # Check if the URL is already embedded in cache
        if request.url in embedding_cache:
            chunks, chunk_embeddings = embedding_cache[request.url]
        else:
            # Fetch and split webpage content
            page_content = fetch_page_content(request.url)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(page_content)

            # Initialize OpenAI components dynamically with API key
            embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # Compute embeddings once and cache
            chunk_embeddings = embedding_model.embed_documents(chunks)
            embedding_cache[request.url] = (chunks, chunk_embeddings)

        # Compute query embedding
        embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Reinitialize for consistency
        query_embedding = embedding_model.embed_query(request.question)

        # Retrieve relevant chunks
        relevant_chunks = get_most_relevant_chunks(query_embedding, chunk_embeddings, chunks)

        # Generate response
        context = "\n".join(relevant_chunks)
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        response = llm.invoke(f"Answer the question based on this context:\n\n{context}\n\nQuestion: {request.question}")

        # Extract answer
        response_text = response.content if hasattr(response, "content") else str(response)

        return {"answer": response_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize_page/")
async def summarize_page(request: SummarizationRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Missing or invalid API key")

    openai_api_key = authorization.split("Bearer ")[1].strip()

    style_prompts = {
        "Concise": "Summarize the following text in a short and precise way.",
        "Bullet Points": "Summarize the following text using bullet points.",
        "Casual": "Summarize the following text in a friendly and conversational way.",
        "Professional": "Summarize the following text in a formal and professional manner."
    }

    if request.style not in style_prompts:
        raise HTTPException(status_code=400, detail="Invalid summarization style")

    try:
        page_content = fetch_page_content(request.url)
        if not page_content:
            raise HTTPException(status_code=400, detail="Failed to extract webpage content")

        # **Step 1: Split text into smaller chunks**
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_text(page_content)

        # **Step 2: Initialize OpenAI Chat Model**
        llm = ChatOpenAI(openai_api_key=openai_api_key)

        # **Step 3: Summarize each chunk separately**
        summaries = []
        for chunk in chunks[:5]:  # Process up to 5 chunks to avoid token overload
            prompt = f"{style_prompts[request.style]}\n\nText:\n{chunk}"
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)
            summaries.append(response_text)

        # **Step 4: Combine summaries into a final response**
        final_summary = "\n".join(summaries)

        return {"summary": final_summary}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/get_suggestions/")
# async def get_suggestions(request: AnalyzeRequest, authorization: str = Header(None)):
#     print("1")
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=400, detail="Missing or invalid API key")

#     openai_api_key = authorization.split("Bearer ")[1].strip()

#     try:
#         llm = ChatOpenAI(openai_api_key=openai_api_key)
#         page_content = fetch_page_content(request.url)
#         if not page_content:
#             raise HTTPException(status_code=400, detail="Failed to extract webpage content")
#         # prompt = f"""
#         # Extract key entities from the following text:        
#         # "{page_content}"        
#         # Provide a JSON response with categories like "people", "organizations", "locations", "events", and "concepts".
#         # Please make sure the entities are most important topic as per the documents and maximum entries of every entities should be less than 5.
#         # """

#         prompt = f"""
#         Task: Extract the four most important entities from the following text:
#         "{page_content}"
#         Instructions:
#         Identify the four key entities (people, organizations, locations, dates, concepts, or other critical elements) that best represent the core meaning of the text.
#         Ensure that the selected entities capture the essence of the content.
#         Construct a coherent and concise phrase (4 to 6 words) incorporating these four entities in a meaningful way.
#         Make sure to Provide a JSON response of phrase
#         Example Input:
#         "Hey there! If you're looking to get started with LangGraph, we've got all the resources you need right here. From basic tutorials to deployment guides and FAQs, we've got you covered. Wondering if LangGraph is open source or if it impacts your app's performance? We've got answers to those questions too. And if you're curious about the differences between LangGraph and other frameworks, we can help with that too. So go ahead, dive in and start exploring all that LangGraph has to offer!
#         Hey there! If you're wondering about using LangGraph, here are some common questions and answers to help you out. LangChain isn't necessary to use LangGraph - they're separate tools. LangGraph stands out from other agent frameworks, and it shouldn't slow down your app's performance. LangGraph is open source and free to use, and there's also a LangGraph Platform available. You have deployment options for the platform, and there's an API reference and troubleshooting section to help you out. Hope this info helps!
#         LangGraph is an open-source orchestration framework that is more low-level and controllable than LangChain. It won't slow down your app and is great for complex tasks tailored to your company's needs. LangGraph Platform is a service for deploying and scaling LangGraph applications, with different deployment options including self-hosted, cloud, and bring your own cloud. Plus, it's all free to use!
#         Hey there! The LangGraph Platform is a cool tool that you can deploy on your own infrastructure. While it's not open source, there is a free self-hosted version available with basic features. The Cloud SaaS deployment option is free during beta, but will eventually become a paid service. Don't worry, they'll give plenty of notice before charging. You can also opt for the Bring Your Own Cloud or Self-Hosted Enterprise options, but those are paid services. If you want more info, reach out to their sales team. And good news - LangGraph works with any LLMs, even if they don't support tool calling. So go ahead and have fun using it with OSS LLMs too! Plus, you can use LangGraph Studio without logging into LangSmith if you want. Happy exploring!
#         Hey there! Just a heads up, we use cookies to make your experience on our site better. By accepting them, you help us improve our documentation. Thanks a bunch! Also, we use Google Analytics and GitHub to keep track of things and make sure everything is running smoothly. Thanks for your understanding! ❤️"

#         Expected Output:
#         Extracted Entities: LangGraph, LangChain, LLM, AI capabilities
#         Framed Phrase: "
#         LangGraph vs LangChain differences
#         LangGraph Platform deployment options
#         LangGraph open source license
#         LangGraph OSS LLM compatibility
#         LangGraph Studio local setup"
#         """
#         response = llm.invoke([HumanMessage(content=prompt)])   
#         print (response)     
#         return {"suggestions": response.content}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_suggestions/")
async def get_suggestions(request: AnalyzeRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Missing or invalid API key")

    openai_api_key = authorization.split("Bearer ")[1].strip()

    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        page_content = fetch_page_content(request.url)
        
        if not page_content:
            raise HTTPException(status_code=400, detail="Failed to extract webpage content")

        prompt = f"""
        Task: Extract the four most important entities from the following text:
        "{page_content}"

        Instructions:
        1. Identify the four key entities (people, organizations, locations, dates, concepts, or other critical elements) that best represent the core meaning of the text.
        2. Ensure that the selected entities capture the essence of the content and should have menaing also avoid any special characters.
        3. Construct a **concise and meaningful phrase** (4 to 6 words) using these four entities.
        4. **Return the result in a structured JSON format.**

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
        """

        response = llm.invoke([HumanMessage(content=prompt)])

        # Ensure the response is valid JSON
        try:
            suggestions = json.loads(response.content)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse response from OpenAI.")

        #print(suggestions)
        return {"suggestions": suggestions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))