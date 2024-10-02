from langchain_chroma import Chroma
from elasticsearch import Elasticsearch, helpers
import numpy as np
from tqdm import tqdm
from openai import OpenAI  # Ensure you have a compatible OpenAI client
import time

# === Configuration ===
# Paths and settings
chroma_persist_directory = r"C:\Users\jamie\OneDrive\Documents\Chatbot\SelfQueryRag\UserGuideRAG\file_vectorstore"
elasticsearch_host = "http://localhost:9200"
index_name = "vector_store"
embedding_dim = 768  # Adjust based on your embeddings
batch_size = 1000

# === Initialize Elasticsearch ===
es = Elasticsearch(
    elasticsearch_host,
    verify_certs=False,  # Set to True if using valid SSL certificates
    basic_auth=("elastic", "password")
)

# === Create Elasticsearch Index ===
mapping = {
    "mappings": {
        "properties": {
            "document_id": {"type": "keyword"},
            "content": {"type": "text"},
            "metadata": {"type": "object"},
            "embedding": {
                "type": "dense_vector",
                "dims": embedding_dim,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)
    print(f"Index '{index_name}' created.")
else:
    print(f"Index '{index_name}' already exists.")

# === Initialize OpenAI Client for Embedding Generation ===
# Make sure to `pip install openai` first
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embedding(text, model="model-identifier"):
    """
    Generates an embedding for the given text using the specified model.
    
    Args:
        text (str): The input text to embed.
        model (str): The identifier of the embedding model to use.
        
    Returns:
        list: The embedding vector.
    """
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text[:30]}... | Error: {e}")
        # Optionally, implement retry logic or return a default embedding
        return None

# === Load Chroma Vectorstore ===
chroma_vectorstore = Chroma(persist_directory=chroma_persist_directory)

# === Retrieve Data from Chroma ===
chroma_data = chroma_vectorstore._collection.get()
documents = chroma_data['documents']
metadatas = chroma_data['metadatas']
ids = chroma_data['ids']

print(f"Retrieved {len(documents)} documents from Chroma.")

# === Prepare and Insert Data into Elasticsearch ===

def normalize_embedding(embedding):
    """
    Normalizes the embedding vector to unit length.

    Args:
        embedding (list or np.array): The embedding vector.

    Returns:
        list: The normalized embedding vector.
    """
    if embedding is None:
        return [0.0] * embedding_dim  # Return a zero vector or handle as needed
    embedding_np = np.array(embedding)
    norm = np.linalg.norm(embedding_np)
    if norm == 0:
        return embedding  # Return the original embedding if norm is zero
    normalized = (embedding_np / norm).tolist()
    return normalized

actions = []
failed_embeddings = 0  # Counter for failed embedding generations

for doc_id, content, metadata in tqdm(zip(ids, documents, metadatas), total=len(documents), desc="Preparing documents"):
    # Generate embedding using the LLM
    embedding = get_embedding(content)
    
    if embedding is None:
        failed_embeddings += 1
        # Optionally, skip this document or use a default embedding
        continue  # Skipping for this example
    
    # Normalize embedding if using cosine similarity
    normalized_embedding = normalize_embedding(embedding)
    
    action = {
        "_index": index_name,
        "_id": doc_id,  # Ensure unique IDs
        "_source": {
            "document_id": doc_id,
            "content": content,
            "metadata": metadata,
            "embedding": normalized_embedding
        }
    }
    actions.append(action)

    # Bulk insert in batches of `batch_size` to optimize performance
    if len(actions) >= batch_size:
        try:
            helpers.bulk(es, actions)
            actions = []
        except Exception as e:
            print(f"Error during bulk insertion: {e}")
            # Optionally, implement retry logic or log failed actions

# Insert any remaining documents
if actions:
    try:
        helpers.bulk(es, actions)
    except Exception as e:
        print(f"Error during final bulk insertion: {e}")

print(f"Data migration to Elasticsearch completed with {failed_embeddings} failed embeddings.")

# === Sample Search ===
# Generate embedding for a sample query
sample_text = "I want to define an atenna"
sample_embedding = get_embedding(sample_text)
normalized_sample_embedding = normalize_embedding(sample_embedding)

query = {
    "knn": {
        "field": "embedding",  # Name of your dense_vector field
        "query_vector": normalized_sample_embedding,  # The query embedding
        "k": 5,  # Number of nearest neighbors to return
        "num_candidates": 100  # Number of candidate vectors to consider for higher accuracy
    }
}

try:
    response = es.search(index=index_name, body=query)
    print("Search Results:")
    print("=========")
    for hit in response['hits']['hits']:
        print(f"Score: {hit['_score']}, Content: {hit['_source']['metadata']}")
except Exception as e:
    print(f"Error during search: {e}")
