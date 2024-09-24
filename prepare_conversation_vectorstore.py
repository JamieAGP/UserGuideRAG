#!/usr/bin/env python3
"""
update_vectorstore_with_conversations.py

This script processes synthetic conversation JSON files and updates the existing Chroma vector store by:
1. Extracting context from the system prompts.
2. Generating metadata (section titles, tags, descriptions, example questions) for each context chunk.
3. Creating embeddings for each chunk.
4. Adding the enriched chunks to the existing Chroma vector store.

Ensure that you have the necessary dependencies installed and that your OpenAI client is properly configured.
"""

import os
import json
import uuid
from pathlib import Path
from functools import lru_cache

from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document


# ------------------------------
# Configuration Classes
# ------------------------------

class Config:
    def __init__(
        self,
        base_dir: Path = Path(__file__).parent,
        conversation_files_dir: Path = None,
        conversation_vectorstore_dir: Path = None,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF",
        llm_model: str = "ggml-model-Q8_0.gguf",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        openai_base_url: str = "http://localhost:1234/v1",
        openai_api_key: str = "lm-studio"
    ):
        self.base_dir = base_dir
        self.conversation_files_dir = conversation_files_dir or self.base_dir / 'conversation_files'
        self.conversation_vectorstore_dir = conversation_vectorstore_dir or self.base_dir / 'conversation_vectorstore'
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key

# ------------------------------
# Custom Classes
# ------------------------------

class CustomEmbeddings:
    def __init__(self, get_embedding_func):
        self.get_embedding = get_embedding_func

    def embed_documents(self, texts):
        return [self.get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        return self.get_embedding(text)

class CallableLLM:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def __call__(self, prompt, system_prompt=None):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=150,  # Adjust as needed
            stream=False  # Disable streaming for batch processing
        )
        return response.choices[0].text.strip()

# ------------------------------
# Utility Functions
# ------------------------------

def get_embedding(text: str, client: OpenAI, model: str) -> list:
    cleaned_text = text.replace("\n", " ")
    embedding_response = client.embeddings.create(input=[cleaned_text], model=model)
    return embedding_response.data[0].embedding

def load_jsonl(file_path):
    """
    Loads a JSONL file and returns a list of JSON objects.
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
    return data_list    

def process_conversation_files(config: Config, client: OpenAI) -> list:
    """
    Processes conversation JSON files and returns a list of Document objects.
    """
    # Initialize the language model callable
    llm_callable = CallableLLM(client, config.llm_model)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, 
        chunk_overlap=config.chunk_overlap
    )

    # List to hold all enriched documents
    enriched_docs = []

    # Ensure that the conversation_files directory exists
    if not config.conversation_files_dir.exists():
        raise FileNotFoundError(f"The directory {config.conversation_files_dir} does not exist.")

    # Iterate over all JSON files in the conversation_files directory
    for file_path in config.conversation_files_dir.glob('*.jsonl'):
        print(f"Processing conversation file: {file_path.name}")
        
        conversation_data = load_jsonl(file_path)
        total_conversations = len(conversation_data)
        
        # Process each conversation
        for idx, conversation in enumerate(conversation_data):
            conversation = conversation['conversations']

            # Extract the system message
            system_message = None
            user_prompts = []
            responses = []
            for message in conversation:
                if message['from'] == 'system':
                    system_message = message['value']
                elif message['from'] == 'human':
                    user_prompts.append(message['value'])
                elif message['from'] == "gpt":
                    responses.append(message['value'])

            if system_message:
                # Extract the context from the system message
                context = extract_context_from_system_message(system_message)

                additional_metadata = {
                    "user_prompts": "+".join(user_prompts),
                    "responses": "+".join(responses),
                    "conversation_id": str(uuid.uuid4())
                }

                # Create Document object
                doc = Document(
                    page_content=context,
                    metadata={
                        "source": file_path.name,
                        **additional_metadata
                    }
                )
                enriched_docs.append(doc)     
                print(f"Processed conversation {idx+1}/{total_conversations}")              
            else:
                print(f"No system message found in conversation in file {file_path.name}")
    return enriched_docs

def extract_context_from_system_message(system_message: str) -> str:
    """
    Extracts the context from the system message.
    """
    # Assuming the system message contains 'Context information is below:'
    parts = system_message.split('Context information is below:')
    if len(parts) > 1:
        context = parts[1].strip()
    else:
        # If the split didn't work, use the entire message
        context = system_message.strip()
    return context

def main(
    config: Config = Config(),
    client: OpenAI = None
):
    """
    Main function to execute the data preparation and vector store update.
    """
    # Initialize OpenAI client if not provided
    if client is None:
        client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)

    print("Starting vector store update with conversation files...")

    # Load and process the conversation documents
    enriched_docs = process_conversation_files(config, client)
    print(f"Total enriched conversation documents: {len(enriched_docs)}")

    # Instantiate the custom embeddings
    embeddings = CustomEmbeddings(lambda text: get_embedding(text, client, config.embedding_model))

    # Check if vector store already exists
    if config.conversation_vectorstore_dir.exists():
        print(f"Vector store already exists at {config.conversation_vectorstore_dir}. Loading existing vector store.")
        vectorstore = Chroma(
            persist_directory=str(config.conversation_vectorstore_dir),
            embedding_function=embeddings
        )
        # Add new documents to the vector store
        vectorstore.add_documents(enriched_docs)
        print(f"Added {len(enriched_docs)} new documents to the vector store.")
    else:
        vectorstore = Chroma.from_documents(
            documents=enriched_docs,
            embedding=embeddings,
            persist_directory=str(config.conversation_vectorstore_dir)
        )
        print(f"Vector store created and saved at {config.conversation_vectorstore_dir}")
        return

    print("Vector store update completed successfully.")

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    main()
