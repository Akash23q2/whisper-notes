# /workspace/whisper-notes/whisper-notes-dev/app/routes/rag_routes.py
from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from app.services.rag_pipeline import RagPipeline
from app.services.rag_service import data_injestion,query_engine


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

# --- Pydantic Models ---

class AddDocumentResponse(BaseModel):
    document_ids: List[str]
    message: str

class SearchRequest(BaseModel):
    query: str
    collection_name: str = "learning_notes"
    k: int = Field(4, gt=0, description="Number of documents to return")

class Document(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[Document]

# --- API Endpoints ---

@rag_router.post("/api/rag/add", status_code=201)
async def add_document(
    collection_name: str = Form("learning_notes"),
    description:str=Form('describe content'),
    text: Optional[str] = Form(None, description="Raw text to embed."),
    pdf_url: Optional[str] = Form(None, description="PDF URL to embed."),
    file: Optional[UploadFile] = File(None, description="Upload a file to embed.")
):
    """
    Add a document to a RAG collection.
    Only one of `text`, `pdf_url`, or `file` should be provided.
    """
    sources = [text, pdf_url, file and file.filename]
    if sum(bool(s) for s in sources) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one of 'text', 'pdf_url', or 'file'."
        )

    try:
        if text:
            logger.info(f"Adding text document to collection: {collection_name}")
            rag_model=data_injestion(text_content=text, collection_name=collection_name,description=description)
        elif pdf_url:
            logger.info(f"Adding PDF URL to collection: {collection_name}")
            rag_model=data_injestion(pdf_url=pdf_url, collection_name=collection_name,description=description)
        elif file:
            logger.info(f"Adding file '{file.filename}' to collection: {collection_name}")
            await file.seek(0)
            rag_model=data_injestion(pdf_path=file.file, collection_name=collection_name,description=description)

        return {"message": "Document added and embedded successfully."}

    except Exception as e:
        logger.error(f"Failed to add document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Failed to add document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/api/rag/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for documents in a collection using a query.
    """
    try:
        logger.info(f"Searching '{request.collection_name}' for: '{request.query}'")
        search_results = query_engine(
            query=request.query,
            collection_name=request.collection_name,
            n_results=request.k
        )

        # Convert each result into a Document
        results = []
        for doc in search_results:
            # If your search result includes metadata, unpack it
            if isinstance(doc, dict):
                results.append(Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {})))
            else:  # if it's just a string
                results.append(Document(page_content=doc, metadata={}))

        return SearchResponse(query=request.query, results=results)

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
