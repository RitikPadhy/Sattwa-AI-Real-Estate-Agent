# session_manager.py

class ModelSession:
    def __init__(self):
        self.history = []

    def add_to_history(self, prompt, response):
        self.history.append({"prompt": prompt, "response": response})

    def get_previous_data(self):
        return self.history

    def clear_history(self):
        self.history = []
