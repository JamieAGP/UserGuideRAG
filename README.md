# VectorStore Management and RAG Inference Pipeline

This project automates the preparation, access, and inference on Chroma vector stores, which are used in a Retrieval-Augmented Generation (RAG) system. It efficiently handles document and conversation data processing, embedding generation, and vector store management, providing a streamlined pipeline for information retrieval and response generation.

## Features

- **VectorStore Preparation**:
  - **Conversation Data**: Processes synthetic conversation files in the "sharegpt" conversation format for supervised fine-tuning (https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/conversation.html), storing them in a conversation vector store.
  - **File Data**: Splits text documents into chunks, generates metadata (section titles, keywords, example questions, etc), and stores enriched data into a file vector store.

- **Metadata Vectorisation**:
  - Retrieves documents and metadata from both the conversation and file vector stores for filtering use.

- **Inference System**:
  - Implements a RAG pipeline that:
    - Loads preprocessed vector stores.
    - Retrieves relevant documents based on user queries.
    - Generates responses using an AI language model loaded on LM studio. Easily adaptable for use of an API.
  
## How It Works

1. **VectorStore Preparation**: The conversation and file data are processed to generate embeddings and metadata. These are stored in Chroma vector stores for efficient retrieval.
   
2. **RAG Inference**: The pipeline retrieves documents using semantic search from vector stores and generates responses based on user queries.

3. **Main Workflow**: The `main.py` script orchestrates the preparation, access, and inference, ensuring all vector stores are ready and up to date.

## Limitations

- Metadata is prepared and vectorised, then accessed, put into a separate file and vectorised again
    - Could probably skip this process and access straight from initial vectorstore
- Limited for prompts requiring distinct pieces of context
    - Should change the retriver from a basic similarity retriever to a multi-query or self query retriever


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repo/vectorstore-rag-pipeline.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
Load an inference model and an embedding model (nomic-ai/nomic-embed-text is good) into LM studio and start the local HTTP server.

To run the entire workflow, including vector store preparation, metadata preparation, and inference:

```bash
python main.py
```

To run just the inference if vectorstores are in order:

```bash
python inference.py
```
