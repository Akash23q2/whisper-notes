## imports ##
from app.services.rag_pipeline import RagPipeline
import re
import unicodedata

## methods ##
def data_injestion(pdf_path=None,pdf_url=None,text_content=None,
                   collection_name="default_collection", chunks=None,chunksize=500,
                   chunk_overlap=50,batch_size=32,
                   embedding_model:str=None,
                   db_path:str=None):
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
    return rag_model

def query_engine(rag_model:RagPipeline,query,collection_name="default_collection",pretty_print=True,
                 n_results=5,db_path=None):
    collection = rag_model.client.get_or_create_collection(name=collection_name)
    query_emb=rag_model.embedder.encode(rag_model._make_chunks(query)).tolist()
    results = collection.query(query_embeddings=query_emb,n_results=n_results)
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