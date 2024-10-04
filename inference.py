#!/usr/bin/env python3
"""
inference.py

This module handles the inference phase of the RAG pipeline by:
1. Connecting to the Elasticsearch vector store.
2. Retrieving relevant documents based on user queries using KNN search.
3. Generating responses using the language model.
4. Collecting and logging user feedback.

You can import this module and call the `main` function from another script.

Ensure that you have the necessary dependencies installed and that the vector store has been prepared.
"""

import os
import json
from pathlib import Path
import csv
from datetime import datetime, timezone
import tempfile
import numpy as np

# Import necessary libraries from langchain and other dependencies
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

# Elasticsearch imports
from elasticsearch import Elasticsearch

# ===========================
# Configuration and Setup
# ===========================

class Config:
    """Configuration class holding all constant values."""
    BASE_DIR = Path(__file__).parent
    FEEDBACK_LOG_PATH = BASE_DIR / 'feedback_log.csv'

    EMBEDDING_MODEL = "model-identifier"  # Replace with your actual embedding model identifier
    LLM_MODEL = "ggml-model-Q8_0.gguf"

    OPENAI_BASE_URL = "http://localhost:1234/v1"
    OPENAI_API_KEY = "lm-studio"

    PROMPT_TEMPLATE = """You are a technical support AI for the software package 'Visualyse Professional Version 7'. You have specialised knowledge on how to perform simulations of 
a range of radiocommunication systems within the software. Use your knowledge to help questioners perform their task.
The following context, in triple backticks, is taken from Visualyse Professional information sources and **may** help you answer the question from the Visualyse Professional User at the end.
Think step by step how to answer the question. If you do not know how to respond to the question/prompt at the end from the provided context, tell the user. The context of the question/prompt will **always** be about Visualyse Professional. 

CONTEXT:

```{context}```

PROMPT: {question}

ACCURATE ANSWER:"""

    # Elasticsearch settings
    ELASTICSEARCH_HOST = "http://localhost:9200"
    ELASTICSEARCH_USERNAME = "elastic"
    ELASTICSEARCH_PASSWORD = "password"
    INDEX_NAME = "vector_store"  # The name of your Elasticsearch index

# Initialize the OpenAI client
client = OpenAI(base_url=Config.OPENAI_BASE_URL, api_key=Config.OPENAI_API_KEY)

# ===========================
# Helper Classes
# ===========================

class CallableLLM:
    """
    Wrapper class to make the language model callable.
    """
    def __init__(self, client, model=Config.LLM_MODEL):
        self.client = client
        self.model = model

    def __call__(self, prompt, system_prompt=None):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=0.3,
            max_tokens=512,  
            stream=True  
        )
        for chunk in response:
            choice = chunk.choices[0].text
            yield choice

# ===========================
# Helper Functions
# ===========================

def get_embedding(text, model=Config.EMBEDDING_MODEL):
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

def normalize_embedding(embedding):
    """
    Normalizes the embedding vector to unit length.

    Args:
        embedding (list or np.array): The embedding vector.

    Returns:
        list: The normalized embedding vector.
    """
    if embedding is None:
        return [0.0] * len(embedding)  # Return a zero vector or handle as needed
    embedding_np = np.array(embedding)
    norm = np.linalg.norm(embedding_np)
    if norm == 0:
        return embedding  # Return the original embedding if norm is zero
    normalized = (embedding_np / norm).tolist()
    return normalized

def retrieve_relevant_docs(user_question, es_client, index_name):
    """
    Retrieves relevant documents from Elasticsearch using KNN search.

    Args:
        user_question (str): The user's input question.
        es_client (Elasticsearch): Elasticsearch client instance.
        index_name (str): The name of the Elasticsearch index.

    Returns:
        list: A list of retrieved Document objects.
    """
    # Get embedding for the question
    sample_embedding = get_embedding(user_question)
    normalized_sample_embedding = normalize_embedding(sample_embedding)
    # Prepare the query
    query = {
        "size": 5,
        "query": {
            "knn": {
                "field": "embedding",
                "query_vector": normalized_sample_embedding,
                "k": 3,
                "num_candidates": 100
            }
        }
    }
    # Perform the search
    response = es_client.search(index=index_name, body=query)
    # Extract the documents
    retrieved_docs = []
    for hit in response['hits']['hits']:
        content = hit['_source']['content']
        metadata = hit['_source']['metadata']
        doc = Document(page_content=content, metadata=metadata)
        retrieved_docs.append(doc)
    return retrieved_docs

def format_docs(docs):
    """
    Formats retrieved documents into a single context string and logs it to a temporary file.

    Args:
        docs (list): List of Document objects.

    Returns:
        str: The concatenated context string.
    """
    context = []
    for doc in docs:
        doc_context_string = (
            f"=================\n"
            f"Source: {doc.metadata.get('source', 'N/A')}\n"
            f"Section title: {doc.metadata.get('section', 'N/A')}\n"
            f"Description: {doc.metadata.get('description', 'N/A')}\n"
            f"Content: {doc.page_content}\n\n"
        )
        context.append(doc_context_string)
    context = "\n\n".join(context)

    # Create a temporary file for logging purposes
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding="utf-8", suffix='.txt') as temp_file:
        temp_file.write(context)
        temp_file_path = temp_file.name

    print(f"\033[91mSelected Context has been written to a temporary file: {temp_file_path}\033[0m")
    return context

def collect_feedback():
    """
    Prompts the user to provide feedback on the generated response.

    Returns:
        str: The user's feedback ('yes' or 'no').
    """
    while True:
        feedback = input("\n\nWas this answer useful? (yes/no): ").strip().lower()
        if feedback in ['yes', 'no']:
            return feedback
        else:
            print("Please enter 'yes' or 'no'.")

def log_feedback(question, response, retrieved_docs, feedback, feedback_log_path=Config.FEEDBACK_LOG_PATH):
    """
    Logs user feedback along with the question and retrieved document metadata.

    Args:
        question (str): The user's question.
        response (str): The generated response.
        retrieved_docs (list): List of retrieved Document objects.
        feedback (str): The user's feedback ('yes' or 'no').
        feedback_log_path (Path): Path to the feedback log CSV file.
    """
    # Ensure the feedback log file exists and has headers
    if not feedback_log_path.exists():
        with open(feedback_log_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'question', 'response', 'retrieved_docs_metadata', 'feedback'])
    
    with open(feedback_log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Convert metadata to string representation
        metadata_str = str([doc.metadata for doc in retrieved_docs])
        writer.writerow([datetime.now(timezone.utc).isoformat(), question, response, metadata_str, feedback])

# ===========================
# Main Execution Function
# ===========================

def main():
    """
    Main function to handle user interactions and generate responses.
    """
    # Initialize Elasticsearch client
    es_client = Elasticsearch(
        Config.ELASTICSEARCH_HOST,
        verify_certs=False,  # Set to True if using valid SSL certificates
        basic_auth=(Config.ELASTICSEARCH_USERNAME, Config.ELASTICSEARCH_PASSWORD)
    )
    index_name = Config.INDEX_NAME  # The name of your Elasticsearch index

    custom_rag_prompt = PromptTemplate.from_template(Config.PROMPT_TEMPLATE)

    # Initialize the language model callable
    llm_callable = CallableLLM(client)

    print("\033[92mRAG Inference System Ready. Type 'exit' or 'quit' or ctrl+c to end.\033[0m")

    while True:
        try:
            # Prompt user for input
            question = input("\033[94m\nEnter your prompt: \033[0m").strip()
            
            if question.lower() in ['exit', 'quit']:
                print("Exiting the RAG Inference System. Goodbye!")
                break

            if not question:
                print("Please enter a valid question.")
                continue

            # Retrieve relevant documents
            retrieved_docs = retrieve_relevant_docs(question, es_client, index_name)
            
            if not retrieved_docs:
                print("Sorry, I couldn't find any relevant information to answer your question.")
                continue

            # Format the retrieved documents
            context = format_docs(retrieved_docs)

            # Construct the prompt
            prompt = custom_rag_prompt.format(context=context, question=question)

            # Generate the response
            print("\033[94m\nResponse: \033[0m", end='', flush=True)
            response_text = ""
            for chunk in llm_callable(prompt, system_prompt=None):
                print(chunk, end='', flush=True)  # Display each chunk as it arrives
                response_text += chunk

            # Collect and log feedback
            feedback = collect_feedback()
            log_feedback(question, response_text, retrieved_docs, feedback)

        except KeyboardInterrupt:
            print("\nExiting the RAG Inference System.")
            break

if __name__ == "__main__":
    main()
