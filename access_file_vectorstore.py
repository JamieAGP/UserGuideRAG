from langchain_chroma import Chroma
from collections import Counter
import json

class CustomEmbeddings:
    """
    Custom embedding class to generate embeddings for documents and queries.
    """
    def embed_documents(self, texts):
        pass
    
    def embed_query(self, text):
        pass

def initialize_vector_store(vectorstore_folderpath='vectorstore'):
    """Initialize the Chroma vector store with custom embeddings."""
    embeddings = CustomEmbeddings()
    vectorstore = Chroma(
        persist_directory=vectorstore_folderpath,  
        embedding_function=embeddings
    )
    return vectorstore

def retrieve_documents(vectorstore):
    """Retrieve all documents from the vector store."""
    return vectorstore.get()

def process_metadata(all_metadata):
    """Process and structure metadata from documents."""
    meta_data = {}

    for doc in all_metadata:
        section = doc['section'].strip("*")
        tags = doc['tags'].split("'''")[0].rstrip().split("\n")[0]
        description = doc['description'].split("'''")[0].rstrip()
        example_question = doc['example_question']

        keywords = [tag.strip() for tag in tags.split(",")]

        section_data = {
            "keywords": keywords,
            "description": description,
            "example_question": example_question
        }

        meta_data[section] = section_data
    
    return meta_data

def write_to_json(output_file, data):
    """Write the processed metadata to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def main(vectorstore_folderpath='vectorstore', keyword_filepath='vectorstore_metadata/file_metadata.json'):
    """Main function to execute the workflow."""
    vectorstore = initialize_vector_store(vectorstore_folderpath)
    documents = retrieve_documents(vectorstore)
    all_metadata = documents['metadatas']

    meta_data = process_metadata(all_metadata)
    write_to_json(keyword_filepath, meta_data)

if __name__ == "__main__":
    main(keyword_filepath='vectorstore_metadata/file_metadata.json')
