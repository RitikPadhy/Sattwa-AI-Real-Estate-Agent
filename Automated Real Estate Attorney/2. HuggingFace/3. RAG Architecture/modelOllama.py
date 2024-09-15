from ollama import OllamaModel
import re
from data_fetching import processed_data
from FAISS_search import search_documents, index_data_to_faiss
from sentence_transformers import SentenceTransformer

# Query templates remain unchanged
query_templates = {
    "contract_drafting": "Draft a {document_type} between {party_A} and {party_B} for {purpose}.",
    "contract_review": "Review the following contract for any legal issues: {contract_text}",
    # Other templates...
}

# Extract arguments function remains the same
def extract_arguments_from_query(user_query):
    arguments = {
        'document_type': re.search(r'(purchase agreement|contract|lease|rental|mortgage|property management)', user_query, re.IGNORECASE),
        'party_A': re.search(r'between (\w+)', user_query),
        'party_B': re.search(r'and (\w+)', user_query),
        'purpose': re.search(r'for (.+?)(?:\.|$)', user_query),
        # Other extractions...
    }
    return {key: value.group(1) if value else None for key, value in arguments.items()}

# Determine the correct template key based on the user query
def determine_template_key(user_query):
    if 'draft' in user_query.lower():
        return "contract_drafting"
    elif 'review' in user_query.lower():
        return "contract_review"
    # Other conditions...
    else:
        return None

# Generate query based on the template
def generate_query(template_key, **kwargs):
    template = query_templates.get(template_key, None)
    if template:
        return template.format(**kwargs)
    return None

# Use OLLAMA model to refine the query
def refine_query_with_ollama(query, model_name):
    ollama_models = {
        'gpt-3.5-turbo': OllamaModel('gpt-3.5-turbo'),
        'llama2': OllamaModel('llama2'),  # Example of using another model like LLaMA
    }
    
    if model_name in ollama_models:
        model = ollama_models[model_name]
        refined_query = model.generate(query, max_tokens=50)
        return refined_query['text']
    else:
        raise ValueError(f"Ollama model '{model_name}' not found")

# Model pipelines for question answering in OLLAMA
qa_models = {
    'gpt-3.5-turbo': OllamaModel('gpt-3.5-turbo'),
    'llama2': OllamaModel('llama2'),
}

# Select OLLAMA model for answering
def select_ollama_model_by_name(model_name):
    if model_name in qa_models:
        return qa_models[model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in the OLLAMA models list")

# Get results from OLLAMA models
def get_results_from_ollama_model(model_name, query, context):
    model = select_ollama_model_by_name(model_name)
    result = model.question_answer(query, context)
    return result['answer']

# Main function handling workflow with OLLAMA models
def main():
    user_query = input("Enter the user query:")
    refine_model_name = input("Enter the model name for refining the query (e.g., 'gpt-3.5-turbo'):")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    index, ids = index_data_to_faiss(processed_data, model)

    search_results = search_documents(user_query, index, processed_data, model, ids)
    context = ' '.join([doc['_source']['text'] for doc in search_results])

    arguments = extract_arguments_from_query(user_query)
    template_key = determine_template_key(user_query)
    generated_query = generate_query(template_key, **arguments) if template_key else user_query

    refined_query = refine_query_with_ollama(generated_query, refine_model_name)

    response_model_name = input("Enter the model name for producing the response (e.g., 'gpt-3.5-turbo'):")
    result = get_results_from_ollama_model(response_model_name, refined_query, context)

    clean_response = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', result)
    print(f"Refining Model: {refine_model_name}")
    print(f"Response Model: {response_model_name}")
    print(f"Answer: {clean_response}")

if __name__ == "__main__":
    main()
