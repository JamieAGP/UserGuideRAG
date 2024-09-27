# import inference
# import prepare_file_vectorstore import main, Config
# from pathlib import Path
# import access_file_vectorstore

# def main():
#     config = Config(
#         rag_files_dir=Path('RAG_files'),
#         vectorestore_dir=Path('vectorstore'),
#         embedding_model="nomic-ai/nomic-embed-text-v1.5-GGUF",
#         llm_model="ggml-model-Q8_0.gguf",
#         chunk_size=1500,
#         chunk_overlap=200,
#         openai_base_url="http://localhost:1234/v1",
#         openai_api_key="your-api-key"
#     )
#     prepare_file_vectorstore.main(config=config)
#     access_file_vectorstore.main()


# if __name__ == "__main__":
#     main()

import os
from access_conversation_vectorstore import main as access_conversation_main
from access_file_vectorstore import main as access_file_main
from prepare_conversation_vectorstore import main as prepare_conversation_main
from prepare_file_vectorstore import main as prepare_file_main
from inference import main as inference_main

# Define paths for vector store and metadata directories
CONVERSATION_VECTORSTORE_PATH = 'conversation_vectorstore'
FILE_VECTORSTORE_PATH = 'file_vectorstore'
METADATA_PATH = 'vectorstore_metadata'

def check_vectorstore_exists(path):
    """Check if a vectorstore exists by checking if the directory is not empty."""
    return os.path.exists(path) and len(os.listdir(path)) > 0

def ensure_conversation_vectorstore():
    """Ensure the conversation vectorstore is ready."""
    if not check_vectorstore_exists(CONVERSATION_VECTORSTORE_PATH):
        print(f"Conversation vectorstore not found. Preparing conversation vectorstore.")
        prepare_conversation_main()
    else:
        print(f"Conversation vectorstore found. Skipping preparation.")

def ensure_file_vectorstore():
    """Ensure the file vectorstore is ready."""
    if not check_vectorstore_exists(FILE_VECTORSTORE_PATH):
        print(f"File vectorstore not found. Preparing file vectorstore.")
        prepare_file_main()
    else:
        print(f"File vectorstore found. Skipping preparation.")

def main():
    # Ensure both conversation and file vector stores are ready
    ensure_conversation_vectorstore()
    ensure_file_vectorstore()

    # Run access scripts to process the vector stores
    print("Accessing conversation vectorstore...")
    access_conversation_main()

    print("Accessing file vectorstore...")
    access_file_main()

    # Run the inference script
    print("Starting inference...")
    inference_main()

if __name__ == "__main__":
    main()

