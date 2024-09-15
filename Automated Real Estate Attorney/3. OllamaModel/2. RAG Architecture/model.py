import gradio as gr
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from create_embeddings import retriever  # Import the pre-generated retriever
import time  # Import time module to measure processing time

def process_input(question):
    start_time = time.time()  # Start the timer

    # Load local model
    model_local = ChatOllama(model="mistral")  

    # Define the prompt template
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    # Create the chain for retrieval-augmented generation (RAG)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    # Get the response from the model
    response = after_rag_chain.invoke(question)

    end_time = time.time()  # End the timer
    processing_time = end_time - start_time  # Calculate the time taken

    # Return the response along with the processing time
    return f"Response:\n{response}\n\nProcessing Time: {processing_time:.2f} seconds"

# Define Gradio interface which only takes a query and returns a response
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Textbox(label="Enter your question")],
    outputs="text",
    title="Document Query with Ollama",
    description="Enter your question to query the pre-embedded documents."
)

# Launch the interface
iface.launch()
