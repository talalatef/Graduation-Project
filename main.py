from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingestion.document_reader import read_documents
from ingestion.chunker import chunk_text
from ingestion.embedder import Embedder
from ingestion.indexer import Indexer
from retrieval.retriever import Retriever
from synthesis.generator import LLAMAGenerator
import config
import uvicorn


app = FastAPI()

# Ingest documents
documents = read_documents(config.DOCUMENTS_PATH)
chunks = [chunk for doc in documents for chunk in chunk_text(doc)]

# Generate embeddings in batches
embedder = Embedder()
# embeddings = embedder.generate_embeddings(chunks, batch_size=16)

# Index embeddings
# indexer = Indexer()
# indexer.index_embeddings(embeddings)
# indexer.save_index(config.INDEX_FILE_PATH)

# Set up retriever
retriever = Retriever(config.INDEX_FILE_PATH)

# Set up generator
generator = LLAMAGenerator(api_key=config.LLAMA3_API_KEY)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    generated_response: str

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        query_embedding = embedder.generate_embeddings([request.query])[0]
        retrieved_indices = retriever.retrieve(query_embedding)
        retrieved_docs = [chunks[i] for i in retrieved_indices]
        context = " ".join(retrieved_docs)
        
        generated_response = generator.generate(context, request.query)

        return QueryResponse(generated_response=generated_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
