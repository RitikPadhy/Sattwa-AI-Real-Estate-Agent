from transformers import pipeline
import re   # Used to represent regular expressions
from data_fetching import processed_data
from FAISS_search import search_documents, index_data_to_faiss
from sentence_transformers import SentenceTransformer

query_templates = {
    # Contract Drafting and Review
    "contract_drafting": "Draft a {document_type} between {party_A} and {party_B} for {purpose}.",
    "contract_review": "Review the following contract for any legal issues: {contract_text}",
    "contract_summarization": "Summarize the key points of the following contract: {contract_text}",

    # Due Diligence
    "due_diligence_property": "Perform due diligence for purchasing a property at {address}. Provide a list of potential legal issues.",
    "due_diligence_liens": "Check for any liens or encumbrances on the property at {address}.",

    # Legal Advice
    "legal_term_explanation": "Explain the implications of the following legal term: {term}.",
    "dispute_resolution": "Advice on resolving a {dispute_type} between {party_A} and {party_B}.",
    "compliance_requirements": "What are the compliance requirements for {action} in {location}?",

    # Transaction Management
    "transaction_closure_steps": "Outline the steps required to close a real estate transaction.",
    "closing_documents_checklist": "Prepare a checklist of closing documents needed for property sale.",

    # Legal Document Preparation
    "legal_document_preparation": "Prepare a {document_type} for {party_A} to act on behalf of {party_B}.",
    "standard_form_filling": "Fill out a {form_type} form with the following details: {details}."
}

# Extract aguments from the user query
def extract_arguments_from_query(user_query):
    arguments = {
        'document_type': re.search(r'(purchase agreement|contract|lease|rental|mortgage|property management)', user_query, re.IGNORECASE),
        'party_A': re.search(r'between (\w+)', user_query),
        'party_B': re.search(r'and (\w+)', user_query),
        'purpose': re.search(r'for (.+?)(?:\.|$)', user_query),  # Adjusted to match multiple words
        'term': re.search(r'explain the term (\w+)', user_query, re.IGNORECASE),
        'dispute_type': re.search(r'resolving a (\w+)', user_query, re.IGNORECASE),
        'address': re.search(r'at (.+)', user_query, re.IGNORECASE),
        'action': re.search(r'compliance requirements for (\w+)', user_query, re.IGNORECASE),
        'location': re.search(r'in (\w+)', user_query, re.IGNORECASE),
        'form_type': re.search(r'fill out a (\w+)', user_query, re.IGNORECASE)
    }
    # Cleans and return all the matched groups
    return {key: value.group(1) if value else None for key, value in arguments.items()}

def determine_template_key(user_query):
    # Determine the correct template for each type
    if 'draft' in user_query.lower():
        return "contract_drafting"
    elif 'review' in user_query.lower():
        return "contract_review"
    elif 'summarize' in user_query.lower():
        return "contract_summarization"
    elif 'due diligence' in user_query.lower():
        return "due_diligence_property"
    elif 'liens' in user_query.lower():
        return "due_diligence_liens"
    elif 'explain' in user_query.lower():
        return "legal_term_explanation"
    elif 'resolve' in user_query.lower():
        return "dispute_resolution"
    elif 'compliance' in user_query.lower():
        return "compliance_requirements"
    elif 'transaction' in user_query.lower():
        return "transaction_closure_steps"
    elif 'checklist' in user_query.lower():
        return "closing_documents_checklist"
    elif 'prepare' in user_query.lower():
        return "legal_document_preparation"
    elif 'fill out' in user_query.lower():
        return "standard_form_filling"
    else:
        return None

# Generate the generic query based on these arguments
def generate_query(template_key, **kwargs):
    template = query_templates.get(template_key, None)
    if template:
        return template.format(**kwargs)
    return None

# Refine the generated query for our model to understand the query even better
def refine_query(query, model_name):
    text_gen_models = {
        'gpt-3.5-turbo': pipeline('text-generation', model='gpt-3.5-turbo'),
        't5-small': pipeline('text-generation', model='t5-small'),
    }
    if model_name in text_gen_models:
        nlp = text_gen_models[model_name]
        refined_query = nlp(query, max_length=50, num_return_sequences=1)[0]['generated_text']
        return refined_query
    else:
        raise ValueError(f"Text generation model '{model_name}' not found")

# Model pipelines
qa_models = {
    'bert-large-uncased-whole-word-masking-finetuned-squad': pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad'),
    'bert-base-uncased': pipeline('question-answering', model='bert-base-uncased'),
    'roberta-base-squad2': pipeline('question-answering', model='deepset/roberta-base-squad2'),
    'roberta-large': pipeline('question-answering', model='roberta-large'),
    'albert-base-v2': pipeline('question-answering', model='albert-base-v2'),
    'albert-xxlarge-v2': pipeline('question-answering', model='albert-xxlarge-v2'),
    't5-small': pipeline('question-answering', model='t5-small'),
    't5-large': pipeline('question-answering', model='t5-large'),
    'deberta-base': pipeline('question-answering', model='microsoft/deberta-base'),
    'distilbert-base-cased-distilled-squad': pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
}

# Function to select a model based on the model name
def select_model_by_name(model_name):
    if model_name in qa_models:
        return qa_models[model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in the models list")

# Generate the refined query and get results from the selected model
def get_results_from_models(model_name, query, context):
    model_pipeline = select_model_by_name(model_name)
    result = model_pipeline(question=query, context=context)
    return result

# Defining main function to take care of the workflow
def main():
    # Get the user query
    user_query = input("Enter the user query:")

    # Which model would you want to use in order to get your response
    refine_model_name = input("Enter the model name for refining the query:")

    # Load the pre-trained model for embeddings and find out the index and ids
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    index, ids = index_data_to_faiss(processed_data, model)

    # Search for relevant documents
    search_results = search_documents(user_query, index, processed_data, model, ids)

    # Combine context from search results
    context = ' '.join([doc['_source']['text'] for doc in search_results])

    # Process the query for arguments
    arguments = extract_arguments_from_query(user_query)

    # Get the template key for the generated query
    template_key = determine_template_key(user_query)

    # Generate the query with the new arguments
    generated_query = generate_query(template_key, **arguments) if template_key else user_query

    # Refine the generated query for each model to understand
    refined_query = refine_query(generated_query, refine_model_name)

    # Which model would you want to use in order to get your response
    response_model_name = input("Enter the model name for producing the response:")

    # Get the result from the selected model
    result = get_results_from_models(response_model_name, refined_query, context)

    # Get the cleaner response
    clean_response = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', result['answer'])

    # Print the model, result and score. Score here is the confidence the model has in its answer.
    print(f"Refining Model: {refine_model_name}")
    print(f"Response Model: {response_model_name}")
    print(f"Answer: {clean_response}")
    print(f"Score: {result.get('score', 'N/A')}")

if __name__ == "__main__":
    main() 