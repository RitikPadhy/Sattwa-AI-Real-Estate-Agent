# app.py

from flask import Flask, request, jsonify
from model_handler import ModelHandler

app = Flask(__name__)
model_handler = ModelHandler()

@app.route('/ask', methods=['POST'])
def ask_model():
    data = request.get_json()
    prompt = data.get('prompt')
    
    # Generate response using the model and session context
    response = model_handler.generate_response(prompt)
    
    return jsonify({"response": response})

@app.route('/clear_session', methods=['POST'])
def clear_session():
    model_handler.clear_session()
    return jsonify({"message": "Session cleared."})

if __name__ == '__main__':
    app.run(debug=True)
