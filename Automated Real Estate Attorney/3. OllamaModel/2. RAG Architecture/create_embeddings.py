from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from data_processing import get_preprocessed_data
from langchain.schema import Document  # Import Document class

# Function to create embeddings and store them in ChromaDB
def embed_documents():
    # Get the preprocessed data from MongoDB and Excel
    data = get_preprocessed_data()

    # Convert the preprocessed data to Document objects (required by LangChain)
    docs = [Document(page_content=item['data'], metadata={"url": item['url'], "collection": item['collection']}) for item in data]
    
    # Split the documents into chunks for better embeddings
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs)

    # Store embeddings in Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="real-estate-law-chroma",
        embedding=OllamaEmbeddings(
            model="nomic-embed-text",
        )
    )

    # Return the retriever from vector store
    return vectorstore.as_retriever()

# Create embeddings and store retriever
retriever = embed_documents()

# Now you can use the retriever for querying the documents
