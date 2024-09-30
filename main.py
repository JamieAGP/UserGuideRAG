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
        input("Conversation vectorstore not found. Ensure a comprehension model is loaded to prepare vectorstore. Hit Enter to continue...")
        prepare_conversation_main()
        return False
    else:
        print(f"Conversation vectorstore found. Skipping preparation.")
        return True

def ensure_file_vectorstore():
    """Ensure the file vectorstore is ready."""
    if not check_vectorstore_exists(FILE_VECTORSTORE_PATH):
        input("File vectorstore not found. Ensure a comprehension model is loaded to prepare vectorstore. Hit Enter to continue...")
        prepare_file_main()
        return False
    else:
        print(f"File vectorstore found. Skipping preparation.")
        return True


def main():
    # Ensure both conversation and file vector stores are ready
    conversation_vectorestore = ensure_conversation_vectorstore()
    file_vectorestore = ensure_file_vectorstore()

    if not conversation_vectorestore or not file_vectorestore:
        input ("Ensure wanted inference model is loaded for interaction. Hit Enter to continue")

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

