# model_handler.py

from session_manager import ModelSession
from model import generate  # Importing from model.py
from data_fetching import fetch_data_from_source  # Integration
from data_collection import collect_data  # Integration
from FAISS_search import perform_faiss_search  # Integration

class ModelHandler:
    def __init__(self):
        self.session = ModelSession()

    def generate_response(self, prompt):
        previous_responses = self.session.get_previous_data()
        
        # Incorporate previous data into the prompt if needed
        full_prompt = self._construct_prompt(prompt, previous_responses)
        
        # Fetch additional data if required
        fetched_data = fetch_data_from_source(prompt)
        collected_data = collect_data(prompt)
        search_result = perform_faiss_search(fetched_data['data'])  # Example integration
        
        # Combine everything into the final prompt
        final_prompt = f"{full_prompt}\n\n{fetched_data['data']}\n\n{collected_data}\n\n{search_result}"
        
        # Generate response using the model
        response = generate(final_prompt)
        
        # Add to session history
        self.session.add_to_history(prompt, response)
        
        return response

    def _construct_prompt(self, current_prompt, previous_data):
        # Construct the prompt by appending previous context
        if previous_data:
            previous_responses = [entry["response"] for entry in previous_data]
            context = "\n".join(previous_responses)
            return f"{context}\n\n{current_prompt}"
        else:
            return current_prompt

    def clear_session(self):
        self.session.clear_history()
