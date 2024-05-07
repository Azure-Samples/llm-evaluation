import json, os, requests
import pandas as pd
from dotenv import load_dotenv
from src.generate_synthetic_data import SyntheticDataGenerator

load_dotenv(override=True)

def get_sample_index_data(top=10):
    payload = json.dumps({
    "count": True,
    "search": "*",
    "searchFields": "synopsis",
    "select": "id, title, synopsis",
    "top": top
    })

    headers = {
    'Content-Type': 'application/json',
    'api-key': os.getenv("AZURE_AI_SEARCH_KEY")
    }

    response = requests.request("POST", os.getenv("AZURE_AI_SEARCH_SERVICE"), headers=headers, data=payload)
    return json.loads(response.text)['value']

if __name__ == '__main__':
    # Initialize the synthetic data generator
    generator = SyntheticDataGenerator()
    
    # Get the sample index data
    data = get_sample_index_data()
    
    # Generate a QA evaluation dataset
    data = generator.generate_qa_evaluation_dataset(data, "synopsis")
    
    # Export to jsonl file (for using on Azure AI Studio)
    output_file = "data\movies-generated-qa-EN-US.jsonl"
    data_df = pd.DataFrame(data, columns=list(data.keys()))
    data_df.to_json(output_file, lines=True, orient="records")
    
    print("QA evaluation dataset generated successfully.")