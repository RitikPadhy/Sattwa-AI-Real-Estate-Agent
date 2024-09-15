# This is the basic understanding of how the app works without the UI

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import sqlite3

model_local = ChatOllama(model="mistral")       # Defines our model

# 1. Split data into chunks
urls = [
    "https://ollama.com/",
    "https://ollama.com/blog/windows-preview",
    "https://ollama.com/blog/openai-compatibility",
]
docs = [WebBaseLoader(url).load() for url in urls]      # Extracts all the data from the URLS
docs_list = [item for sublist in docs for item in sublist]      # Combines all the data together
# Function for splitting the data into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)     
doc_splits = text_splitter.split_documents(docs_list)       # Splits the documents into smaller text sections using chunk size and chunk overlap

# 2. Convert documents to Embeddings and store them
# Creates a chroma vector store which accepts a list of document chunks to be embedded into the vector store. rag-chroma is our vectorstore.
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(
         model="nomic-embed-text",
    )
)
# Converts vector store into retriever which will find documents in the vector store which are most similiar to the query based on embeddings
retriever = vectorstore.as_retriever()  

# 3. Before RAG
print("Before RAG\n")   
before_rag_template = "What is {topic}" 
before_rag_prompt = ChatPromptTemplate.from_template(before_rag_template)       # Used for creating dynamic questions based on the topic
# We give it our initial prompt template and local model. Then, StrOutputParser takes the output and parses it into a string format.
before_rag_chain = before_rag_prompt | model_local | StrOutputParser()  
print(before_rag_chain.invoke({"topic": "Ollama"}))         # Giving topic as Ollama

# 4. After RAG
print("\n########\nAfter RAG\n")
# Here our RAG template, the only difference is that we have introduced context and Question
after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
# Then we call our chaing giving in our context and question
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)
print(after_rag_chain.invoke("What is Ollama?"))

