import os, json
from src.azure_index_manager import AzureIndexManager
from src.generate_synthetic_data import SyntheticDataGenerator
from azure.search.documents.indexes.models import (
    SearchFieldDataType,
    SearchableField,
    SearchField,
    SimpleField,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
    HnswParameters,
    VectorSearchAlgorithmKind,
    ExhaustiveKnnVectorSearchAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField
)
from concurrent.futures import ThreadPoolExecutor

def get_movies_synthetic_data(generate_new_data=False):
    if(generate_new_data):
        # Create an instance of SyntheticDataGenerator
        generator = SyntheticDataGenerator()

        system_message = "You are an AI assistant that helps to create datasets. \
                    You will generate a dataset of movies with the following columns: title, synopsis, genre, director, and release year. \
                    The synopsis should be a long description of the movie. \
                    The dataset should have 30 rows and MUST be in a json format."
        
        user_prompt = "Generate a dataset of movies with the following columns: title, synopsis, genre, director, and release year. \
                    Do not repeat any movie. The dataset should have at least 30 rows and each movie must be unique in the dataset"
        
        # Generate synthetic data
        synthetic_data = generator.generate_dataset(system_message, user_prompt)
        
        # Remove duplicated records based on the title field
        synthetic_data = clean_data(synthetic_data)
        
        # Persist json dataset to a file
        with open("data/movies_EN-US.json", "w") as file:
            file.write(json.dumps(synthetic_data))
    else:
        synthetic_data = load_synthetic_data('movies_EN-US.json')

    return synthetic_data

# Generate embeddings for the movies dataset. We use ThreadPoolExecutor to parallelize the process.
def generate_embeddings(index_manager, dataset):
    json_data=[]
    id=0
    
    def process_item(item):
        nonlocal id
        json_data.append({
            "id": str(id),
            "title": str(item['title']),
            "synopsis": item['synopsis'],
            "genre": item['genre'],
            "director": item['director'],
            "release_year": item['release_year'],
            "synopsisVector": index_manager.generate_embeddings(item['synopsis']),
            "genreVector": index_manager.generate_embeddings(item['genre'])
        })
        id+=1

    with ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(process_item, dataset)
       
    # Output embeddings to docVectors.json file
    # with open("data\docVectors-movies-EN-US.json", "w") as f:
    #     json.dump(json_data, f)
    
    return json_data
    
def load_synthetic_data(filename):
    # Load the synthetic data
    with open(filename, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset

def create_movies_index(index_name):
    # Create an instance of AzureIndexManager
    index_manager = AzureIndexManager()

    # Define the fields for the search index
    fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String,  analyzer_name='en.microsoft', sortable=True, filterable=True, facetable=False),
    SearchableField(name="synopsis", type=SearchFieldDataType.String,  analyzer_name='en.microsoft', sortable=True, filterable=True, facetable=False),
    SearchableField(name="genre", type=SearchFieldDataType.String, analyzer_name='en.microsoft', sortable=True, filterable=True, facetable=False),
    SearchableField(name="director", type=SearchFieldDataType.String, analyzer_name='en.microsoft', sortable=True, filterable=True, facetable=False),
    SimpleField(name="release_year", type=SearchFieldDataType.Int32),
    SearchField(name="synopsisVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile="myHnswProfile"),
    SearchField(name="genreVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_profile="myHnswProfile"),
    ]
    
    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswVectorSearchAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric="cosine"
                )
            ),
            ExhaustiveKnnVectorSearchAlgorithmConfiguration(
                name="myExhaustiveKnn",
                kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                parameters=ExhaustiveKnnParameters(
                    metric="cosine"
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm="myHnsw",
                vectorizer="myOpenAI"
            ),
            VectorSearchProfile(
                name="myExhaustiveKnnProfile",
                algorithm="myExhaustiveKnn",
                vectorizer="myOpenAI"
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                name='myOpenAI',
                kind="azureOpenAI",
                azure_open_ai_parameters=AzureOpenAIParameters(
                    resource_uri=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    deployment_id=os.getenv("ADA_EMBEDDING_NAME"),
                    api_key=os.getenv("AZURE_OPENAI_KEY")
                )
        )  
    ]  

    )

    # Configure the semantic settings
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=PrioritizedFields(
            title_field=SemanticField(field_name="title"),
            prioritized_keywords_fields=[SemanticField(field_name="genre")],
            prioritized_content_fields=[SemanticField(field_name="synopsis")]
        )
    )
    
    # Create the index
    index_manager.create_index(index_name, fields, vector_search, semantic_config)

def clean_data(dataset):
    # Extract the generated dataset from the API response
    dataset = json.loads(dataset)['movies']

    # Remove duplicated records based on the title field
    unique_dataset = []
    titles = set()

    for movie in dataset:
        title = movie['title']
        if title not in titles:
            unique_dataset.append(movie)
            titles.add(title)
    
    return unique_dataset

if __name__ == "__main__":    
    index_manager = AzureIndexManager()
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    
    try:
        # Create an index for the movies dataset
        create_movies_index(index_name)
        
        # Get the movies dataset (you can set generate_new_data to True to generatenew  synthetic data)
        movies_dataset = get_movies_synthetic_data(generate_new_data=False)
        
        # Generate embeddings for the movies dataset
        movies_dataset = generate_embeddings(index_manager, movies_dataset)
        
        # Ingest the movies embeddings dataset to the Azure AI index
        index_manager.ingest_index_data(movies_dataset)
        
        print('Index created and loaded successfully!')
    except Exception as e:
        print(f'Index process failed! Error: {e}')