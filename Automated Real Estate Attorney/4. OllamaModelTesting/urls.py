from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter

def embed_documents():
    # Predefined URLs
    urls = [
        "https://mohua.gov.in/upload/uploadfiles/files/Real_Estate_Act_2016(2).pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/2338/1/A1882-04.pdf",
        "https://www.indiacode.nic.in/bitstream/123456789/15937/1/the_registration_act%2C1908.pdf",
    ]

    # Load documents from URLs
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    # Store embeddings in Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(
            model="nomic-embed-text",
        )
    )

    # Return the retriever from vector store
    return vectorstore.as_retriever()

# Create embeddings and store retriever
retriever = embed_documents()
