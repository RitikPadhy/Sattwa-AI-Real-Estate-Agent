import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import fitz 
from io import BytesIO
import certifi
import re

# Load data from Excel file
file_name = 'data.xlsx'  # Ensure data.xlsx is in the same directory as this script
data = pd.read_excel(file_name)

# Set up MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['real-estate-law']

# Define collections
collections = ['contracts', 'regulations', 'case_law', 'property_records', 'transactional_documents', 'public_records', 'legal_forms']

# Function for creating collections in the database
def create_collections():
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)

# Function to fetch data from URL based on CSS selectors
def fetch_data(url, css_selectors):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    extracted_data = ""
    for selector in css_selectors:
        if selector.strip():  # Check if the selector is not empty
            elements = soup.select(selector)
            if not elements:
                print(f"No elements found for selector: {selector}")
            for element in elements:
                extracted_data += element.get_text(separator="\n").strip() + "\n"
    return extracted_data

# Function to fetch data from an online PDF URL
def fetch_pdf_data(url):
    response = requests.get(url)
    pdf_document = fitz.open(stream=BytesIO(response.content))
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

# Create collections if they do not exist
create_collections()

# Iterate through each row of the dataframe and store data in MongoDB
for index, row in data.iterrows():
    if row['Problem'] == 'No':
        if row['Extracted'] == 'No':
            collection_name = row['Collection']
            topic = row['Topic']
            quality = row['Quality']
            url = row['URL']  
            countries = row['Countries Involved']
            country_source = row['Country of Source']
                       
            if row['PDF URL'].lower() == 'yes':
                extracted_data = fetch_pdf_data(url)
                document = {
                    'topic': topic,
                    'quality': quality,
                    'source': url,
                    'countries': countries,
                    'country_source': country_source,
                    'data': extracted_data,
                }

                collection = db[collection_name]
                collection.insert_one(document)
            
            elif row['PDF URL'].lower() == 'no':
                if row['Manual Scraping'].lower() == 'no':
                    css_selectors = [selector.strip() for selector in row['CSS Selectors'].split(';') if selector.strip()]
                    extracted_data = fetch_data(url, css_selectors)

                    if extracted_data != "":
                        document = {
                            'topic': topic,
                            'quality': quality,
                            'source': url,
                            'countries': countries,
                            'country_source': country_source,
                            'data': extracted_data,
                        }

                        collection = db[collection_name]
                        collection.insert_one(document)

                elif row['Manual Scraping'].lower() == 'yes':
                    extracted_data = ""
                    document = {
                        'topic': topic,
                        'quality': quality,
                        'source': url,
                        'countries': countries,
                        'country_source': country_source,
                        'data': extracted_data,
                    }

                    collection = db[collection_name]
                    collection.insert_one(document)
