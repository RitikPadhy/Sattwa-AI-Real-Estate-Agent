import pandas as pd
from pymongo import MongoClient
import re

# MongoDB Collection
client = MongoClient('mongodb://localhost:27017/')
db = client['real-estate-law']

# Read excel sheet
excel_path = "../2. MongoDB data/data.xlsx"  
excel_data = pd.read_excel(excel_path)

# Create a dictionary from the Excel sheet 
excel_dict = excel_data.set_index('URL')['Cleaned'].to_dict()

# Fetch data from all collection
def fetch_data_from_all_documents(db, excel_dict):
    data = []
    collections = db.list_collection_names()
    for collection in collections:
        documents = db[collection].find()
        for doc in documents:
            url = doc.get('source')
            cleaned = excel_dict.get(url, 'No')
            data.append({
                'url': url,
                'collection': collection,
                'topic': doc.get('topic'),
                'quality': doc.get('quality'),
                'countries_involved': doc.get('countries'),
                'country_of_source': doc.get('country_source'),
                'cleaned': cleaned,
                'data': doc.get('data')
            })
    
    return data

# Preprocess data
def preprocess_data(data):
    processed_data = []
    for item in data:
        if item.get('cleaned') == 'No':
            text = item.get('data', '')

            # Normalize the text to lowercase
            text = text.lower()

            # Replace newlines (\n) with spaces
            text = text.replace('\n', ' ')
            
            # Remove unnecessary punctuation but keep parentheses and numbers
            text = re.sub(r'[^\w\s\(\)\.,]', ' ', text)
            
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text).strip()
            
            
            # Update the cleaned text in the item
            item['data'] = text
        processed_data.append(item)
    
    return processed_data

# Specify the topic to fetch
data = fetch_data_from_all_documents(db, excel_dict)
processed_data = preprocess_data(data)
print(processed_data)
