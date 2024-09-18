#!/usr/bin/env python3
"""
inference.py

This script handles the inference phase of the RAG pipeline by:
1. Loading the preprocessed Chroma vector store.
2. Retrieving relevant documents based on user queries.
3. Generating responses using the language model.
4. Collecting and logging user feedback.

Ensure that you have the necessary dependencies installed and that the vector store has been prepared using 'prepare_vectorstore.py'.
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
VECTORESTORE_DIR = BASE_DIR / 'vectorstore'
FEEDBACK_LOG_PATH = BASE_DIR / 'feedback_log.csv'

# Define embedding model
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"

# Define LLM model for response generation
LLM_MODEL = "ggml-model-Q8_0.gguf"

# Define chunking parameters (should match prepare_vectorstore.py)
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
            max_tokens=512,  # Adjust as needed
            stream=False  # Disable streaming for synchronous processing
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

def retrieve_with_metadata(query, retriever, metadata_filters=None, top_k=5):
    """
    Retrieves documents based on semantic similarity and optional metadata filters.

    Args:
        query (str): The user's question.
        retriever (Retriever): The retriever object from the vector store.
        metadata_filters (dict, optional): Metadata key-value pairs to filter the documents.
        top_k (int): Number of top documents to retrieve.

    Returns:
        list: A list of retrieved Document objects.
    """
    # Perform semantic search
    results = retriever.get_relevant_documents(query, k=top_k)

    if metadata_filters:
        # Filter results based on metadata
        filtered_results = [
            doc for doc in results
            if all(doc.metadata.get(k) == v for k, v in metadata_filters.items())
        ]
        return filtered_results if filtered_results else results  # Fallback if no filter matches

    return results

def rank_chunks_with_metadata(retrieved_docs, metadata_boost=None):
    """
    Ranks retrieved documents based on their similarity scores and metadata.

    Args:
        retrieved_docs (list): List of retrieved Document objects.
        metadata_boost (dict, optional): Metadata key-value pairs to boost document scores.

    Returns:
        list: Ranked list of Document objects.
    """
    ranked_docs = []
    for doc in retrieved_docs:
        score = getattr(doc, 'score', 0)  # Assuming 'score' attribute exists
        if metadata_boost:
            for key, value in metadata_boost.items():
                if doc.metadata.get(key) == value:
                    score += 0.1  # Adjust boost value as needed
        ranked_docs.append((doc, score))
    
    # Sort documents based on the boosted score
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked_docs]

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

def collect_feedback():
    """
    Prompts the user to provide feedback on the generated response.

    Returns:
        str: The user's feedback ('yes' or 'no').
    """
    while True:
        feedback = input("Was this answer helpful? (yes/no): ").strip().lower()
        if feedback in ['yes', 'no']:
            return feedback
        else:
            print("Please enter 'yes' or 'no'.")

def log_feedback(question, retrieved_docs, feedback, feedback_log_path=FEEDBACK_LOG_PATH):
    """
    Logs user feedback along with the question and retrieved document metadata.

    Args:
        question (str): The user's question.
        retrieved_docs (list): List of retrieved Document objects.
        feedback (str): The user's feedback ('yes' or 'no').
        feedback_log_path (Path): Path to the feedback log CSV file.
    """
    # Ensure the feedback log file exists and has headers
    if not feedback_log_path.exists():
        with open(feedback_log_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'question', 'retrieved_docs_metadata', 'feedback'])
    
    with open(feedback_log_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Convert metadata to string representation
        metadata_str = str([doc.metadata for doc in retrieved_docs])
        writer.writerow([datetime.utcnow().isoformat(), question, metadata_str, feedback])

# ===========================
# Load the Vector Store
# ===========================

def load_vectorstore():
    """
    Loads the Chroma vector store from the specified directory.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    if not VECTORESTORE_DIR.exists():
        raise FileNotFoundError(f"The vector store directory {VECTORESTORE_DIR} does not exist. Please run 'prepare_vectorstore.py' first.")
    
    embeddings = CustomEmbeddings()
    vectorstore = Chroma(
        persist_directory=str(VECTORESTORE_DIR),
        embedding_function=embeddings,
    )
    return vectorstore

# ===========================
# Define the Prompt Template
# ===========================

template = """The following context, in triple backticks, is taken from the Visualyse Professional User Guide and should help you answer the question from the Visualyse Professional User at the end.
Think step by step how to answer the question. If you do not know the answer to the question at the end, tell the user. The context of the question will always be about Visualyse Professional. Always give a response.

CONTEXT:

```{context}```

QUESTION: {question}

ACCURATE ANSWER:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# ===========================
# Define the Inference Function
# ===========================

def retrieve_relevant_docs(user_question, retriever):
    """
    Retrieves and ranks relevant documents based on the user's question.

    Args:
        user_question (str): The user's input question.
        retriever (Retriever): The retriever object from the vector store.

    Returns:
        list: A list of the top-ranked Document objects.
    """
    # Optionally expand the query
    expanded_question = expand_query(user_question)  # Implement if desired

    # Determine metadata filters based on the question
    metadata_filters = determine_metadata_filters(user_question)

    # Retrieve documents using the metadata-aware retrieval
    retrieved_docs = retrieve_with_metadata(
        query=expanded_question,
        retriever=retriever,
        metadata_filters=metadata_filters,
        top_k=5  # Adjust as needed
    )

    # Optionally rank the retrieved documents further
    # retrieved_docs = rank_chunks_with_metadata(retrieved_docs, metadata_boost={"section": "Simulation Basics"})

    # Limit to top-N documents
    TOP_N = 3
    final_retrieved_docs = retrieved_docs[:TOP_N]

    return final_retrieved_docs

def determine_metadata_filters(user_question):
    """
    Analyzes the user's question to determine relevant metadata filters.

    Args:
        user_question (str): The user's input question.

    Returns:
        dict or None: A dictionary of metadata filters or None if no filters apply.
    """
    # Implement logic to extract keywords or topics from the question
    # For simplicity, let's assume we have predefined mappings
    section_keywords = {
        "Simulation Basics": ["simulate", "simulation"],
        "Advanced Topics": ["advanced", "complex"],
        "User Interface": ["interface", "UI", "user interface"],
        "Data Analysis": ["data", "analysis", "reporting"],
        # Add more mappings as needed
    }

    filters = {}
    for section, keywords in section_keywords.items():
        if any(keyword.lower() in user_question.lower() for keyword in keywords):
            filters["section"] = section  # Assuming section titles match exactly
            break  # Use the first matching section

    return filters if filters else None

def expand_query(query):
    """
    Expands the user query using synonym expansion to improve retrieval accuracy.

    Args:
        query (str): The original user query.

    Returns:
        str: The expanded query.
    """
    # Example: Simple synonym expansion (can be enhanced with NLP models)
    synonyms = {
        "simulate": ["model", "emulate", "replicate"],
        "problem": ["issue", "challenge", "difficulty"],
        "configure": ["setup", "adjust", "set up"],
        "optimize": ["improve", "enhance", "refine"],
        # Add more synonyms as needed
    }
    words = query.split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        if word.lower() in synonyms:
            expanded_words.extend(synonyms[word.lower()])
    expanded_query = " ".join(expanded_words)
    return expanded_query

# ===========================
# Main Execution Function
# ===========================

def main():
    """
    Main function to handle user interactions and generate responses.
    """
    print("Loading vector store...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    # Initialize the language model callable
    llm_callable = CallableLLM(client)

    print("\033[92mRAG Inference System Ready. Type 'exit' to quit.\033[0m")

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
            retrieved_docs = retrieve_relevant_docs(question, retriever)

            if not retrieved_docs:
                print("Sorry, I couldn't find any relevant information to answer your question.")
                continue

            # Format the retrieved documents
            context = format_docs(retrieved_docs)

            # Construct the prompt
            prompt = custom_rag_prompt.format(context=context, question=question)

            # Generate the response
            response = llm_callable(prompt, system_prompt=None)  # System prompt is already included in the prompt template

            # Display the response
            print("\033[94m\nResponse: \033[0m", response)

            # Collect and log feedback
            feedback = collect_feedback()
            log_feedback(question, retrieved_docs, feedback)

        except KeyboardInterrupt:
            print("\nExiting the RAG Inference System. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

# ===========================
# Execute the Script
# ===========================

if __name__ == "__main__":
    main()
