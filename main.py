import os
# --- STEP 1: DISABLE PARALLELISM TO PREVENT DATABASE LOCKS ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
import traceback
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from git import Repo

# --- AI & LANGCHAIN IMPORTS ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

app = FastAPI()

# ⚠️ YOUR GROQ KEY
GROQ_API_KEY = "your api key"

# --- FIXED CORS SETTINGS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_PATH = os.path.abspath("./temp_repo")
DB_PATH = os.path.abspath("./chroma_db")
vectorstore = None

class RepoRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"status": "Online", "message": "Groq Backend is running!"}

@app.post("/index")
async def index_repository(request: RepoRequest):
    global vectorstore
    try:
        # 1. FORCE CLOSE DATABASE CONNECTION
        if vectorstore is not None:
            print("🔄 Closing existing database connection...")
            vectorstore = None 
        
        # 2. CLEAR CACHE & WAIT
        # This helps release the file lock on Mac
        import gc
        gc.collect() 
        time.sleep(2) 

        # 3. ROBUST FOLDER DELETION
        # If rmtree fails, we try to rename it (a trick to bypass locks)
        for path in [REPO_PATH, DB_PATH]:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except:
                    # If still locked, move it out of the way
                    os.rename(path, f"{path}_old_{int(time.time())}")

        print(f"📥 Cloning: {request.url}")
        Repo.clone_from(request.url, to_path=REPO_PATH)

        print("🔍 Parsing files...")
        loader = GenericLoader.from_filesystem(
            REPO_PATH, 
            glob="**/*", 
            suffixes=[".py", ".js", ".ts", ".ipynb", ".txt", ".md"],
            parser=LanguageParser() 
        )
        docs = loader.load()
        
        if not docs:
            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            loader = DirectoryLoader(REPO_PATH, glob="**/*.*", loader_cls=TextLoader)
            docs = loader.load()

        # 4. RECREATE DATABASE
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(docs)

        print(f"🧠 Indexing {len(texts)} chunks...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # We create a new client instance every time to avoid reusing locked handles
        vectorstore = Chroma.from_documents(
            documents=texts, 
            embedding=embeddings, 
            persist_directory=DB_PATH
        )
        
        print("✅ Indexing Complete!")
        return {"status": "success", "message": "Successfully indexed!"}
        
    except Exception as e:
        print("❌ BACKEND ERROR:", str(e))
        traceback.print_exc() 
        return {"status": "error", "message": str(e)} # Don't raise 500, return JSON
@app.post("/query")
async def query_codebase(request: QueryRequest):
    global vectorstore
    try:
        # Load from disk if memory is cleared
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if os.path.exists(DB_PATH):
            vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        else:
            return {"answer": "❌ Error: You must index a repository before asking questions."}

        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
        
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), 
            return_source_documents=True
        )
        
        print(f"🧐 Processing Question: {request.question}")
        result = qa.invoke({"question": request.question, "chat_history": []})
        
        return {"answer": result.get("answer", "I couldn't find a specific answer in the code.")}

    except Exception as e:
        print("❌ QUERY ERROR:", str(e))
        traceback.print_exc()
        return {"answer": f"Backend Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)