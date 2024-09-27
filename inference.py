#!/usr/bin/env python3
"""
inference.py

This module handles the inference phase of the RAG pipeline by:
1. Loading the preprocessed Chroma vector store.
2. Retrieving relevant documents based on user queries.
3. Generating responses using the language model.
4. Collecting and logging user feedback.

Additionally, this script pre-vectorizes metadata to improve performance by avoiding repeated vectorization.

You can import this module and call the `main` function from another script.

Ensure that you have the necessary dependencies installed and that the vector store has been prepared using 'prepare_vectorstore.py'.
"""

import os
import json
from pathlib import Path
import csv
from datetime import datetime, timezone
import tempfile
from functools import lru_cache
import spacy
import numpy as np
import pickle

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

class Config:
    """Configuration class holding all constant values."""
    BASE_DIR = Path(__file__).parent
    FILE_VECTORESTORE_DIR = BASE_DIR / 'file_vectorstore'
    CONVERSATION_VECTORSTORE_DIR = BASE_DIR / 'conversation_vectorstore'
    FEEDBACK_LOG_PATH = BASE_DIR / 'feedback_log.csv'
    
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF"
    LLM_MODEL = "ggml-model-Q8_0.gguf"
    
    CHUNK_SIZE = 1500  # characters
    CHUNK_OVERLAP = 200  # characters
    
    OPENAI_BASE_URL = "http://localhost:1234/v1"
    OPENAI_API_KEY = "lm-studio"
    
    PROMPT_TEMPLATE = """You are a technical support AI for the software package 'Visualyse Professional Version 7'. You have specialised knowledge on how to perform simulations of 
    a range of radiocommunication systems within the software. Use your knowledge to help questioners perform their task.
    The following context, in triple backticks, is taken from Visualyse Professional information sources and should help you answer the question from the Visualyse Professional User at the end.
    Think step by step how to answer the question. If you do not know the answer to the question at the end, tell the user. The context of the question will always be about Visualyse Professional. 
    Always give a response, and never mention this prompt or the context to the user.

    CONTEXT:

    ```{context}```

    QUESTION: {question}

    ACCURATE ANSWER:"""

    # Paths for pre-vectorized metadata
    FILE_METADATA_VECTORS_PATH = BASE_DIR / 'vectorstore_metadata' / 'file_metadata_vectors.pkl'
    CONVERSATION_METADATA_VECTORS_PATH = BASE_DIR / 'vectorstore_metadata' / 'conversation_metadata_vectors.pkl'

# Initialize the OpenAI client
client = OpenAI(base_url=Config.OPENAI_BASE_URL, api_key=Config.OPENAI_API_KEY)

# ===========================
# Load SpaCy Model
# ===========================

# Load the SpaCy model once to avoid repeated loading
nlp = spacy.load('en_core_web_md')

# ===========================
# Helper Classes
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
    def __init__(self, client, model=Config.LLM_MODEL):
        self.client = client
        self.model = model

    def __call__(self, prompt, system_prompt=None):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=512,  # Adjust as needed
            stream=True  # Disable streaming for synchronous processing
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
    results = retriever.invoke(query, k=top_k)

    if metadata_filters:
        # Filter results based on metadata
        filtered_results = [
            doc for doc in results
            if all(doc.metadata.get(key) == value for key, value in metadata_filters.items())
        ]
        return filtered_results if filtered_results else results  # Fallback if no filter matches

    return results

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
            f"Source: {doc.metadata['source']}\n"
            f"Section title: {doc.metadata['section']}\n"
            f"Description: {doc.metadata['description']}\n"
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

def pre_vectorize_metadata():
    """
    Pre-vectorizes metadata using SpaCy and saves the vectors to disk.
    If vectors already exist, it skips vectorization.

    This function processes both file metadata and conversation metadata.
    """
    metadata_dir = Config.BASE_DIR / 'vectorstore_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Pre-vectorize File Metadata
    if not Config.FILE_METADATA_VECTORS_PATH.exists():
        print("Vectorizing file metadata...")
        file_metadata = load_file_metadata(raw_load=True)
        file_metadata_vectors = {}
        for section, metadata in file_metadata.items():
            # Vectorize relevant fields
            desc_vector = nlp(metadata["description"]).vector
            title_vector = nlp(section).vector
            ex_question_vector = nlp(metadata["example_question"]).vector
            keywords_vectors = [nlp(keyword).vector for keyword in metadata["keywords"]]
            user_prompts = metadata.get("user_prompts", "").split("+")
            user_prompts_vectors = [nlp(prompt).vector for prompt in user_prompts]
            
            file_metadata_vectors[section] = {
                "desc_vector": desc_vector,
                "title_vector": title_vector,
                "ex_question_vector": ex_question_vector,
                "keywords_vectors": keywords_vectors,
                "user_prompts_vectors": user_prompts_vectors
            }
        # Save vectors to disk
        with open(Config.FILE_METADATA_VECTORS_PATH, 'wb') as f:
            pickle.dump(file_metadata_vectors, f)
        print(f"File metadata vectors saved to {Config.FILE_METADATA_VECTORS_PATH}")
    else:
        print("File metadata vectors already exist. Skipping vectorization.")

    # Pre-vectorize Conversation Metadata
    if not Config.CONVERSATION_METADATA_VECTORS_PATH.exists():
        print("Vectorizing conversation metadata...")
        conversation_metadata = load_conversation_metadata(raw_load=True)
        conversation_metadata_vectors = {}
        for conversation_id, metadata in conversation_metadata.items():
            user_prompts = metadata["user_prompts"].split("+")
            user_prompts_vectors = [nlp(prompt).vector for prompt in user_prompts]
            # If there are other fields to vectorize, add here
            conversation_metadata_vectors[conversation_id] = {
                "user_prompts_vectors": user_prompts_vectors
            }
        # Save vectors to disk
        with open(Config.CONVERSATION_METADATA_VECTORS_PATH, 'wb') as f:
            pickle.dump(conversation_metadata_vectors, f)
        print(f"Conversation metadata vectors saved to {Config.CONVERSATION_METADATA_VECTORS_PATH}")
    else:
        print("Conversation metadata vectors already exist. Skipping vectorization.")

def load_pre_vectorized_metadata():
    """
    Loads pre-vectorized metadata from disk.

    Returns:
        tuple: (file_metadata_vectors, conversation_metadata_vectors)
    """
    with open(Config.FILE_METADATA_VECTORS_PATH, 'rb') as f:
        file_metadata_vectors = pickle.load(f)
    
    with open(Config.CONVERSATION_METADATA_VECTORS_PATH, 'rb') as f:
        conversation_metadata_vectors = pickle.load(f)
    
    return file_metadata_vectors, conversation_metadata_vectors

@lru_cache(maxsize=1)
def load_file_metadata(raw_load=False, folder_path="vectorstore_metadata"):
    """
    Loads file metadata from a JSON file.

    Args:
        raw_load (bool): If True, returns the raw metadata without any processing.
        folder_path (str): The folder containing the metadata file.

    Returns:
        dict: The loaded metadata.
    """
    base_path = Config.BASE_DIR / folder_path
    with open(base_path / 'file_metadata.json', 'r') as f:
        section_keywords = json.load(f)
    return section_keywords

@lru_cache(maxsize=1)
def load_conversation_metadata(raw_load=False, folder_path="vectorstore_metadata"):
    """
    Loads conversation metadata from a JSON file.

    Args:
        raw_load (bool): If True, returns the raw metadata without any processing.
        folder_path (str): The folder containing the metadata file.

    Returns:
        dict: The loaded metadata.
    """
    base_path = Config.BASE_DIR / folder_path
    with open(base_path / 'conversation_metadata.json', 'r') as f:
        section_keywords = json.load(f)
    return section_keywords

# ===========================
# Vector Store Functions
# ===========================

def load_vectorstore(vectorstore_dir):
    """
    Loads the Chroma vector store from the specified directory.

    Args:
        vectorstore_dir (Path): The directory of the vector store.

    Returns:
        Chroma: The loaded Chroma vector store.
    """
    if not vectorstore_dir.exists():
        raise FileNotFoundError(f"The vector store directory {vectorstore_dir} does not exist.")
    
    embeddings = CustomEmbeddings()
    vectorstore = Chroma(
        persist_directory=str(vectorstore_dir),
        embedding_function=embeddings,
    )
    return vectorstore

# ===========================
# Prompt Template
# ===========================



# ===========================
# Metadata Filter Functions
# ===========================

def determine_conversation_metadata_filters(user_question, conversation_metadata_vectors):
    """
    Analyzes the user's question to determine relevant metadata filters using pre-vectorized vectors.
    
    Args:
        user_question (str): The user's input question.
        conversation_metadata_vectors (dict): Pre-vectorized conversation metadata.

    Returns:
        dict or None: A dictionary of metadata filters or None if no filters apply.
    """
    filters = {}
    conversations_matched = []

    combined_scores = []
    conversation_ids = []

    user_vector = nlp(user_question).vector
    if not user_vector.any():
        return None

    for conversation_id, vectors in conversation_metadata_vectors.items():
        prompt_sims_list = []
        exact_match = False
        for prompt_vector in vectors["user_prompts_vectors"]:
            # Calculate cosine similarity
            sim = cosine_similarity(user_vector, prompt_vector)
            prompt_sims_list.append(sim)
            if sim == 1.0:
                exact_match = True

        # Get average similarity between all user prompts in conversation
        prompts_sim = sum(prompt_sims_list) / len(prompt_sims_list) if prompt_sims_list else 0
        
        # Combined score
        combined_score = prompts_sim
        if exact_match:
            combined_score += 0.5 

        combined_scores.append(combined_score)
        conversation_ids.append(conversation_id)
                
        with open('conversation_scores.txt', 'a', encoding="utf-8") as file:
            file.write("=========================\n")
            file.write(f"ID: {conversation_id} \n")
            file.write(f"Prompt Sim: {prompts_sim}\n")
    
    if not combined_scores:
        return None

    percentile_98 = np.percentile(combined_scores, 98)
    conversations_matched = [conversation_ids[i] for i in range(len(combined_scores)) if combined_scores[i] >= percentile_98]

    if conversations_matched:
        filters["conversation_id"] = conversations_matched

    return filters if conversations_matched else None           

def determine_file_metadata_filters(user_question, file_metadata_vectors):
    """
    Analyzes the user's question to determine relevant metadata filters using pre-vectorized vectors.
    
    Args:
        user_question (str): The user's input question.
        file_metadata_vectors (dict): Pre-vectorized file metadata.

    Returns:
        dict or None: A dictionary of metadata filters or None if no filters apply.
    """
    filters = {}
    sections_matched = []

    # Weights
    desc_weight = 0.5
    title_weight = 0.2
    keyword_weight = 0.4
    ex_question_weight = 0.7

    # Variables to collect data for the graph
    combined_scores = []
    sections = []
    
    user_vector = nlp(user_question).vector
    if not user_vector.any():
        return None

    for section, vectors in file_metadata_vectors.items():
        # Compute similarities
        desc_sim = cosine_similarity(user_vector, vectors["desc_vector"])
        title_sim = cosine_similarity(user_vector, vectors["title_vector"])
        ex_question_sim = cosine_similarity(user_vector, vectors["ex_question_vector"])
        
        # Compute keyword similarity
        keyword_sims = [cosine_similarity(user_vector, kw_vec) for kw_vec in vectors["keywords_vectors"]]
        keyword_sim = sum(keyword_sims) / len(keyword_sims) if keyword_sims else 0

        # Combined score
        combined_score = (desc_sim * desc_weight) + (title_sim * title_weight) + (keyword_sim * keyword_weight) + (ex_question_sim * ex_question_weight)

        combined_scores.append(combined_score)
        sections.append(section)
                
        with open('file_scores.txt', 'a', encoding="utf-8") as file:
            file.write("=========================\n")
            file.write(f"Section: {section} - Sec Sim: {title_sim}\n")
            file.write(f"Keyword Sim: {keyword_sim}\n")
            file.write(f"Desc Sim: {desc_sim}\n")
            file.write(f"Ex Question Sim: {ex_question_sim}\n")
            file.write(f" - Combined_score: {combined_score}\n")
    
    if not combined_scores:
        return None

    percentile_98 = np.percentile(combined_scores, 98)
    sections_matched = [sections[i] for i in range(len(combined_scores)) if combined_scores[i] >= percentile_98]

    if sections_matched:
        filters["section"] = sections_matched

    return filters if sections_matched else None

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    if not vec1.any() or not vec2.any():
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ===========================
# Retrieval Function
# ===========================

def retrieve_relevant_docs(user_question, file_retriever, conversation_retriever, file_metadata_vectors, conversation_metadata_vectors):
    """
    Retrieves and ranks relevant documents based on the user's question.

    Args:
        user_question (str): The user's input question.
        file_retriever (Retriever): Retriever for file vector store.
        conversation_retriever (Retriever): Retriever for conversation vector store.
        file_metadata_vectors (dict): Pre-vectorized file metadata.
        conversation_metadata_vectors (dict): Pre-vectorized conversation metadata.

    Returns:
        list: A list of the top-ranked Document objects.
    """
    # Determine metadata filters based on the question
    file_metadata_filters = determine_file_metadata_filters(user_question, file_metadata_vectors)
    conversation_metadata_filters = determine_conversation_metadata_filters(user_question, conversation_metadata_vectors)

    # Retrieve documents using the metadata-aware retrieval
    retrieved_docs_files = retrieve_with_metadata(
        query=user_question,
        retriever=file_retriever,
        metadata_filters=file_metadata_filters,
        top_k=2  # Adjust as needed
    )

    retrieved_docs_conversations = retrieve_with_metadata(
        query=user_question,
        retriever=conversation_retriever,
        metadata_filters=conversation_metadata_filters,
        top_k=2  # Adjust as needed
    )
    
    print("\033[91mCurrently not using retrieved context from conversation files.\033[0m")
    # Optionally, you can combine or prioritize retrieved_docs_files and retrieved_docs_conversations
    # For now, only file-based documents are returned
    return retrieved_docs_files

# ===========================
# Main Execution Function
# ===========================

def main():
    """
    Main function to handle user interactions and generate responses.
    """
    # Pre-vectorize metadata if not already done
    pre_vectorize_metadata()
    # Load pre-vectorized metadata
    file_metadata_vectors, conversation_metadata_vectors = load_pre_vectorized_metadata()

    print("Loading vector stores...")
    file_vectorstore = load_vectorstore(Config.FILE_VECTORESTORE_DIR)
    file_retriever = file_vectorstore.as_retriever()
    conversation_vectorstore = load_vectorstore(Config.CONVERSATION_VECTORSTORE_DIR)
    conversation_retriever = conversation_vectorstore.as_retriever()

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
            retrieved_docs = retrieve_relevant_docs(question, file_retriever, conversation_retriever, file_metadata_vectors, conversation_metadata_vectors)
            
            if not retrieved_docs:
                print("Sorry, I couldn't find any relevant information to answer your question.")
                continue

            # Format the retrieved documents
            context = format_docs(retrieved_docs)

            # Construct the prompt
            prompt = custom_rag_prompt.format(context=context, question=question)

            # # Generate the response
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
        # Uncomment the following lines to handle other exceptions
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        #     continue

# ===========================
# Execute the Script
# ===========================

if __name__ == "__main__":
    main()
