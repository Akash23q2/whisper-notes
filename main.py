from app.services.rag_service import data_injestion, query_engine, delete_data
# import markdown
def main():
    print("Hello from whisper-notes!")
    rag_model=data_injestion(pdf_path='src/Unit3.pdf')
    query=1
    while query:
        query=input("Enter your query: ")
        results=query_engine(rag_model,query=query,n_results=3)
        print(results)


if __name__ == "__main__":
    main()
