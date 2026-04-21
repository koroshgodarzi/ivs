from rag.vectorstore import get_vector_collection

def get_rag_context(user_question: str, n_results: int = 2):
    collection = get_vector_collection()
    
    results = collection.query(
        query_texts=[user_question],
        n_results=n_results
    )

    context_list = results.get("documents", [[]])[0]
    return "\n---\n".join(context_list)