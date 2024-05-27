import os
import openai
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential  
from tenacity import retry, wait_random_exponential, stop_after_attempt  

from dotenv import load_dotenv, find_dotenv
from azure.search.documents.indexes.models import (
    SearchIndex,
    SemanticSearch
)

load_dotenv(find_dotenv(), override=True) 

class AzureIndexManager:
    def __init__(self):
        openai.api_type = "azure"  
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")  
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  
        openai.api_version = "2023-03-15-preview"
        
        self.search_client = SearchClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                                          index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
                                          credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")))
        
        self.index_client = SearchIndexClient(endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"), 
                                              credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")))
        

    def create_index(self, index_name, fields, vector_search_config, semantic_config):
        semantic_settings = SemanticSearch(configurations=[semantic_config])
        
        # Create the search index with the semantic settings
        index = SearchIndex(name=index_name, 
                            fields=fields,
                            vector_search=vector_search_config, 
                            semantic_settings=semantic_settings)

        result = self.index_client.create_or_update_index(index)
        return result        

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    # Function to generate embeddings. We use a retry decorator to handle transient errors.
    def generate_embeddings(self, text):
        response = openai.embeddings.create(
            input=text, model=os.getenv("ADA_EMBEDDING_NAME"))
        embeddings = response.data[0].embedding
        return embeddings

    def ingest_index_data(self, documents, batch_size=500):
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = self.search_client.upload_documents(batch)
            print(f"Uploaded {len(batch)} documents")
        return result
