## imports ##
from app.services.rag_pipeline import RagPipeline
import re
import unicodedata
import chromadb

rag_model=RagPipeline()
global avilable_collections
avilable_collections = {} #dictionary of collection name to description

## methods ##
def data_injestion(pdf_path:str=None,pdf_url:str=None,text_content:str=None,
                   collection_name:str="default_collection",description:str="",
                   chunks=None,chunksize:int=500,
                   chunk_overlap=50,batch_size=32,
                   embedding_model:str=None,
                   db_path:str=None):
    global avilable_collections
    if collection_name in avilable_collections:
        raise ValueError(f"Collection name '{collection_name}' already exists. Please choose a different name.")
    avilable_collections[collection_name]=description
    print("🔍 Final available collections:", avilable_collections)
    rag_model=RagPipeline()
    if pdf_path:
        rag_model.chunks_from_pdf(pdf_path,chunk_overlap=chunk_overlap)
    elif pdf_url:
        rag_model.chunks_from_url(pdf_url,chunk_overlap=chunk_overlap)
    elif text_content:
        rag_model.chunks_from_text(text_content,chunk_overlap=chunk_overlap)
    elif chunks:
        rag_model.chunks=chunks
    else:
        raise ValueError("Please provide either pdf_path, pdf_url, or text_content.")
    
    rag_model.make_embeddings(embedding_model=embedding_model,batch_size=batch_size)
    rag_model.save_embeddings(collection_name=collection_name,db_path=db_path)
    print(f"Data Ingestion completed. Total {len(rag_model.chunks)} chunks created.")

def query_engine(query,collection_name="default_collection",pretty_print=True,
                 n_results=5,db_path=None):
    rag_model.chunks_from_text(text_content=query)
    rag_model.make_embeddings()
    rag_model.client = chromadb.PersistentClient(path=db_path) if db_path else chromadb.Client()
    collection = rag_model.client.get_collection(name=collection_name)
    results = collection.query(query_embeddings=rag_model.embeddings,n_results=n_results)
    return _clean_documents_result(results) if pretty_print else results

def clean_text_response(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'([,;:.!?])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'•{2,}', '•', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'([<>=])\1+', r'\1', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',,', ',', text)
    text = text.strip()
    return text

def _clean_documents_result(result_dict: dict) -> list[str]:
    docs = []
    if "documents" in result_dict and result_dict["documents"]:
        for doc_list in result_dict["documents"]:
            for doc in doc_list:
                cleaned = clean_text_response(doc)
                if cleaned:
                    docs.append(cleaned)
    return docs

    
def delete_data(rag_model,collection_name,db_path):
        rag_model.delete_data(collection_name=collection_name,db_path=db_path)