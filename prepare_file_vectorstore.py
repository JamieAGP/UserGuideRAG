"""
This module processes text files from the 'RAG_files' directory by:
1. Splitting them into chunks.
2. Generating descriptive section titles, keywords, descriptions, and example questions for each chunk using an AI model.
3. Creating embeddings for each chunk.
4. Storing the enriched chunks in a Chroma vector store located in the 'vectorstore' directory.

Ensure that you have the necessary dependencies installed and that your OpenAI client is properly configured.
"""

import os
import json
from pathlib import Path
import csv
from datetime import datetime
import tempfile
from functools import lru_cache

from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document

# ------------------------------
# Configuration Classes (Optional)
# ------------------------------

class Config:
    """
    Configuration class to hold all constant values.
    """
    def __init__(
        self,
        base_dir: Path = Path(__file__).parent,
        rag_files_dir: Path = None,
        vectorestore_dir: Path = None,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5-GGUF",
        llm_model: str = "ggml-model-Q8_0.gguf",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        openai_base_url: str = "http://localhost:1234/v1",
        openai_api_key: str = "lm-studio"
    ):
        self.base_dir = base_dir
        self.rag_files_dir = rag_files_dir or self.base_dir / 'RAG_files'
        self.vectorestore_dir = vectorestore_dir or self.base_dir / 'file_vectorstore'
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
    """
    Custom embedding class to generate embeddings for documents and queries.
    """
    def __init__(self, get_embedding_func):
        self.get_embedding = get_embedding_func

    def embed_documents(self, texts):
        return [self.get_embedding(text) for text in texts]
    
    def embed_query(self, text):
        return self.get_embedding(text)

class CallableLLM:
    """
    Wrapper class to make the language model callable.
    """
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
    """
    Generates an embedding for a given text using the specified model.

    Args:
        text (str): The input text to embed.
        client (OpenAI): The OpenAI client instance.
        model (str): The embedding model to use.

    Returns:
        list: The embedding vector.
    """
    cleaned_text = text.replace("\n", " ")
    embedding_response = client.embeddings.create(input=[cleaned_text], model=model)
    return embedding_response.data[0].embedding

def extract_tags(chunk_content: str, llm_callable: CallableLLM) -> str:
    """
    Extracts jargon or keywords from the chunk content using the language model.

    Args:
        chunk_content (str): The content of the text chunk.
        llm_callable (CallableLLM): The callable language model instance.

    Returns:
        str: A string of extracted tags.
    """
    prompt = (
        "Extract and respond with a list of relevant technical tags or jargon, and meta-descriptive words that indicate the purpose or function of the paragraph (e.g 'definition', 'setup', 'explanation', 'create') from the following text in triple quotes. Your response should contain only the tags.\n\n"
        "For example:\n\n"
        """'''Station Group - Create Empty
        Empty groups can be created and Stations can be added to it from the list of all
        Stations within the simulation. Even if Stations are being created individually it
        can be useful to assign them to a Group – for example to manage simulations
        by collecting like Stations together.
        This also allows groupings of Stations that are not related in any of the ways
        assumed by the other Groups.
        For example, satellite constellations that do not have the symmetries assumed
        by the Constellation Wizard are not excluded in Visualyse. The Constellation
        Wizard can be run several times to produce single planes of satellites with
        differing orbital parameters. Then, by creating and editing an empty Group it is
        possible to put all the planes into the same group and create a new
        constellation.
        Mixed GSO/non-GSO constellations can be created in a similar way'''\n\n"""
        "Tags: Station Group, Like stations, Constellation Wizard, Mixed GSO/non-GSO constellations, example, process, setup\n\n"
        f"'''{chunk_content}'''\n\n"
        "Tags:"
    )
    tags = llm_callable(prompt, system_prompt=None)
    tags = tags.split("'''")[0].rstrip().split("\n")[0]
    # Assuming the tags are comma-separated
    return tags

def extract_description(chunk_content: str, llm_callable: CallableLLM) -> str:
    """
    Extracts a one-sentence description from the chunk content using the language model.

    Args:
        chunk_content (str): The content of the text chunk.
        llm_callable (CallableLLM): The callable language model instance.

    Returns:
        str: A one-sentence description.
    """
    prompt = (
        "Extract and respond with a one sentence description of the following text in triple quotes. Your response should contain only the description.\n\n"
        "For example:\n\n"
        """'''Station Group - Create Empty
Empty groups can be created and Stations can be added to it from the list of all
Stations within the simulation. Even if Stations are being created individually it
can be useful to assign them to a Group – for example to manage simulations
by collecting like Stations together.
This also allows groupings of Stations that are not related in any of the ways
assumed by the other Groups.
For example, satellite constellations that do not have the symmetries assumed
by the Constellation Wizard are not excluded in Visualyse. The Constellation
Wizard can be run several times to produce single planes of satellites with
differing orbital parameters. Then, by creating and editing an empty Group it is
possible to put all the planes into the same group and create a new
constellation.
Mixed GSO/non-GSO constellations can be created in a similar way'''\n\n"""
        "One Sentence Description: Explanation of why creating empty station groups is useful.\n\n"
        f"'''{chunk_content}'''\n\n"
        "One Sentence Description:"
    )
    description = llm_callable(prompt, system_prompt=None)
    
    # Ensure the description is only one sentence
    one_sentence_description = description.split(".")[0] + "."
    one_sentence_description = one_sentence_description.split("'''")[0].rstrip().split("\n")[0]

    return one_sentence_description

def extract_example_question(chunk_content: str, llm_callable: CallableLLM) -> str:
    """
    Extracts a one-sentence question from the chunk content using the language model.

    Args:
        chunk_content (str): The content of the text chunk.
        llm_callable (CallableLLM): The callable language model instance.

    Returns:
        str: A one-sentence question.
    """
    prompt = (
        "Extract and respond with a one sentence question from the point of a user in an AI system, that can be answered with information in the following text in triple quotes. Make the question as overarching as possible to try and cover the main ideas in the paragraph. Your response should contain only the question.\n\n"
        "For example:\n\n"
        """'''Station Group - Create Empty
Empty groups can be created and Stations can be added to it from the list of all
Stations within the simulation. Even if Stations are being created individually it
can be useful to assign them to a Group – for example to manage simulations
by collecting like Stations together.
This also allows groupings of Stations that are not related in any of the ways
assumed by the other Groups.
For example, satellite constellations that do not have the symmetries assumed
by the Constellation Wizard are not excluded in Visualyse. The Constellation
Wizard can be run several times to produce single planes of satellites with
differing orbital parameters. Then, by creating and editing an empty Group it is
possible to put all the planes into the same group and create a new
constellation.
Mixed GSO/non-GSO constellations can be created in a similar way'''\n\n"""
        "One Sentence Question: Why would I want to create an empty station group?\n\n"
        f"'''{chunk_content}'''\n\n"
        "One Sentence Question:"
    )
    question = llm_callable(prompt, system_prompt=None)
    
    # Ensure the question is only one sentence
    one_sentence_question = question.split(".")[0] + "."
    one_sentence_question = one_sentence_question.split("'''")[0].rstrip().split("\n")[0]

    return one_sentence_question

def extract_section_title(chunk_content: str, llm_callable: CallableLLM) -> str:
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
    section_title = section_title.strip().strip("*")
    return section_title

@lru_cache(maxsize=None)
def cached_extract_section_title(chunk_content: str, llm_callable: CallableLLM) -> str:
    """
    Caches the section title extraction to avoid redundant API calls for identical chunks.

    Args:
        chunk_content (str): The content of the text chunk.
        llm_callable (CallableLLM): The callable language model instance.

    Returns:
        str: The generated section title.
    """
    return extract_section_title(chunk_content, llm_callable)

def load_and_process_documents(config: Config, client: OpenAI) -> list:
    """
    Loads text files, splits them into chunks, generates section titles, and enriches them with metadata.

    Args:
        config (Config): Configuration object containing all necessary parameters.
        client (OpenAI): The OpenAI client instance.

    Returns:
        list: A list of enriched Document objects.
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

    # Ensure that the RAG_files directory exists
    if not config.rag_files_dir.exists():
        raise FileNotFoundError(f"The directory {config.rag_files_dir} does not exist.")

    # Iterate over all text files in the RAG_files directory
    for file_path in config.rag_files_dir.glob('*.txt'):
        print(f"Processing file: {file_path.name}")
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_docs = loader.load()

        for doc in loaded_docs:
            # Split the document into chunks
            chunks = text_splitter.split_documents([doc])
            total_chunks = len(chunks)
            for idx, chunk in enumerate(chunks, start=1):
                try:
                    # Generate section title for the chunk
                    section_title = cached_extract_section_title(chunk.page_content, llm_callable)
                except Exception as e:
                    print(f"Error generating section title for chunk {idx} in {file_path.name}: {e}")
                    section_title = "General Information"

                additional_metadata = {
                    "tags": extract_tags(chunk.page_content, llm_callable),
                    "description": extract_description(chunk.page_content, llm_callable),
                    "example_question": extract_example_question(chunk.page_content, llm_callable),
                }

                # Attach metadata
                enriched_doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": file_path.name,
                        "section": section_title,
                        **additional_metadata
                    }
                )
                enriched_docs.append(enriched_doc)
                print(f"Processed chunk {idx}/{total_chunks}")

    return enriched_docs

def main(
    config: Config = Config(),
    client: OpenAI = None
):
    """
    Main function to execute the data preparation and vector store creation.

    Args:
        config (Config, optional): Configuration object. Defaults to a default Config instance.
        client (OpenAI, optional): OpenAI client instance. If None, it will be initialized using config. Defaults to None.
    """
    # Initialize OpenAI client if not provided
    if client is None:
        client = OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)

    print("Starting vector store preparation...")

    # Load and process documents
    enriched_docs = load_and_process_documents(config, client)
    print(f"Total enriched documents: {len(enriched_docs)}")

    # Instantiate the custom embeddings
    embeddings = CustomEmbeddings(lambda text: get_embedding(text, client, config.embedding_model))

    # Check if vector store already exists
    if config.vectorestore_dir.exists():
        print(f"Vector store already exists at {config.vectorestore_dir}. Loading existing vector store.")
        vectorstore = Chroma(
            persist_directory=str(config.vectorestore_dir),
            embedding_function=embeddings
        )
        vectorstore.add_documents(enriched_docs)
        print(f"Added {len(enriched_docs)} new documents to the vector store.")
    else:
        print("Creating a new vector store...")
        vectorstore = Chroma.from_documents(
            documents=enriched_docs,
            embedding=embeddings,
            persist_directory=str(config.vectorestore_dir)
        )
        print(f"Vector store created and saved at {config.vectorestore_dir}")
    
    print("Vector store preparation completed successfully.")

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    main()
