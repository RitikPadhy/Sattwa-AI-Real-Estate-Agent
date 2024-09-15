import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data_fetching import processed_data  # Assumes this contains the data to be indexed

# Generate embeddings and index the data into FAISS
def index_data_to_faiss(data, model, index_path='faiss_index'):
    embeddings = []
    ids = []

    for i, item in enumerate(data):
        embedding = model.encode(item['data'])  # Extracts 'data' field from the data
        embeddings.append(embedding)
        ids.append(i)

    embeddings = np.array(embeddings)

    # Initialize FAISS index
    dim = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)

    # Add vectors to the index
    index.add(embeddings)

    # Optionally, save the index to disk
    faiss.write_index(index, index_path)

    return index, ids

# Search documents using FAISS
def search_documents(user_query, index, processed_data, model, ids):
    # Generate query embedding
    query_embedding = model.encode(user_query).reshape(1, -1)

    # Perform the search
    D, I = index.search(query_embedding, k=5)  # k is the number of top results to retrieve

    search_results = []
    for i in I[0]:
        search_results.append(processed_data[ids[i]])

    return search_results

# if __name__ == "__main__":
#     model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the pre-trained model
    
#     # Index data
#     index, ids = index_data_to_faiss(processed_data, model)

#     # Save the index (optional)
#     # faiss.write_index(index, 'faiss_index')

#     # Search example
#     user_query = "Tell me about RERA"
#     results = search_documents(user_query, index, processed_data, model, ids)
#     print("Search Results:")
#     for result in results:
#         print(result)