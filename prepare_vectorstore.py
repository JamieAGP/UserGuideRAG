#!/usr/bin/env python3
"""
prepare_vectorstore.py

This script processes text files from the 'RAG_files' directory by:
1. Splitting them into chunks.
2. Generating descriptive section titles for each chunk using an AI model.
3. Creating embeddings for each chunk.
4. Storing the enriched chunks in a Chroma vector store located in the 'vectorstore' directory.

Ensure that you have the necessary dependencies installed and that your OpenAI client is properly configured.
"""

import os
from pathlib import Path
import csv
from datetime import datetime
import tempfile
from functools import lru_cache

# Import necessary libraries from langchain and other dependencies
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

# ===========================
# Configuration and Setup
# ===========================

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define paths
BASE_DIR = Path(__file__).parent
RAG_FILES_DIR = BASE_DIR / 'RAG_files'
VECTORESTORE_DIR = BASE_DIR / 'vectorstore'

# Ensure that the RAG_files directory exists
if not RAG_FILES_DIR.exists():
    raise FileNotFoundError(f"The directory {RAG_FILES_DIR} does not exist.")

# Define embedding model
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"

# Define LLM model for section title extraction
LLM_MODEL = "ggml-model-Q8_0.gguf"

# Define chunking parameters
CHUNK_SIZE = 1500  # characters
CHUNK_OVERLAP = 200  # characters

# ===========================
# Define Helper Classes
# ===========================

class CustomEmbeddings:
    """
    Custom embedding class to generate embeddings for documents and queries.
    """
    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        return get_embedding(text)

class CallableLLM:
    """
    Wrapper class to make the language model callable.
    """
    def __init__(self, client, model=LLM_MODEL):
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

# ===========================
# Define Helper Functions
# ===========================

def get_embedding(text, model=EMBEDDING_MODEL):
    """
    Generates an embedding for a given text using the specified model.

    Args:
        text (str): The input text to embed.
        model (str): The embedding model to use.

    Returns:
        list: The embedding vector.
    """
    cleaned_text = text.replace("\n", " ")
    embedding_response = client.embeddings.create(input=[cleaned_text], model=model)
    return embedding_response.data[0].embedding

def extract_section_title(chunk_content, llm_callable):
    """
    Generates a section title for a given text chunk using the language model.

    Args:
        chunk_content (str): The content of the text chunk.
        llm_callable (CallableLLM): The callable language model instance.

    Returns:
        str: The generated section title.
    """
    prompt = (
        "You are an AI assistant tasked with generating concise and descriptive section titles for technical documentation.\n\n"
        "It is imperative that you only respond with the section title, as your response is being processed by an automated script.\n\n"
        "Analyze the following text and provide an appropriate section title that accurately reflects its content.\n\n"
        f"{chunk_content}\n\n"
        "Section Title:"
    )
    section_title = llm_callable(prompt, system_prompt=None)
    return section_title.strip()

@lru_cache(maxsize=None)
def cached_extract_section_title(chunk_content, llm_callable):
    """
    Caches the section title extraction to avoid redundant API calls for identical chunks.

    Args:
        chunk_content (str): The content of the text chunk.
        llm_callable (CallableLLM): The callable language model instance.

    Returns:
        str: The generated section title.
    """
    return extract_section_title(chunk_content, llm_callable)

def format_docs(docs):
    """
    Formats retrieved documents into a single context string and logs it to a temporary file.

    Args:
        docs (list): List of Document objects.

    Returns:
        str: The concatenated context string.
    """
    context = "\n\n".join(doc.page_content for doc in docs)

    # Create a temporary file for logging purposes
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding="utf-8", suffix='.txt') as temp_file:
        temp_file.write(context)
        temp_file_path = temp_file.name

    print(f"\033[91mSelected Context has been written to a temporary file: {temp_file_path}\033[0m")
    return context

def load_and_process_documents():
    """
    Loads text files, splits them into chunks, generates section titles, and enriches them with metadata.

    Returns:
        list: A list of enriched Document objects.
    """
    # Initialize the language model callable
    llm_callable = CallableLLM(client)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # List to hold all enriched documents
    enriched_docs = []

    # Iterate over all text files in the RAG_files directory
    for file_path in RAG_FILES_DIR.glob('*.txt'):
        print(f"Processing file: {file_path.name}")
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            # Split the document into chunks
            chunks = text_splitter.split_documents([doc])
            for idx, chunk in enumerate(chunks, start=1):
                try:
                    # Generate section title for the chunk
                    section_title = cached_extract_section_title(chunk.page_content, llm_callable)
                except Exception as e:
                    print(f"Error generating section title for chunk {idx} in {file_path.name}: {e}")
                    section_title = "General Information"

                

                # Attach metadata
                enriched_doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": file_path.name,
                        "section": section_title
                    }
                )
                enriched_docs.append(enriched_doc)
                print(f"  Processed chunk {idx}: Section Title - {section_title}")

    return enriched_docs

# ===========================
# Main Execution Function
# ===========================

def main():
    """
    Main function to execute the data preparation and vector store creation.
    """
    print("Starting vector store preparation...")

    # Load and process documents
    enriched_docs = load_and_process_documents()
    print(f"Total enriched documents: {len(enriched_docs)}")

    # Instantiate the custom embeddings
    embeddings = CustomEmbeddings()

    # Check if vector store already exists
    if VECTORESTORE_DIR.exists():
        print(f"Vector store already exists at {VECTORESTORE_DIR}. Loading existing vector store.")
        vectorstore = Chroma(
            persist_directory=str(VECTORESTORE_DIR),
            embedding_function=embeddings,
            # metadata_field_info={
            #     "source": "string",
            #     "section": "string"
            # }
        )
    else:
        print("Creating a new vector store...")
        vectorstore = Chroma.from_documents(
            documents=enriched_docs,
            embedding=embeddings,
            persist_directory=str(VECTORESTORE_DIR),
            # metadata_field_info={
            #     "source": "string",
            #     "section": "string"
            # }
        )
        print(f"Vector store created and saved at {VECTORESTORE_DIR}")

    print("Vector store preparation completed successfully.")

if __name__ == "__main__":
    main()
