from pathlib import Path
import chromadb
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize the client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define the custom embedding function
def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

# Create a custom embedding class
class CustomEmbeddings:
    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)

# Wrapper class to make the client callable
class CallableLLM:
    def __init__(self, client, model="text-davinci-003"):
        self.client = client
        self.model = model # Dummy model
    
    def __call__(self, prompt):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=200
        )
        print(f"RAW RESPONSE: {response}")
        return response.choices[0].text.strip()

# Load and prepare the single text file
file_path = Path(__file__).parent / 'User Guide.txt'  # Replace with your actual file name
loader = TextLoader(file_path, encoding="utf-8")
doc_data = loader.load()

# Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
splits = text_splitter.split_documents(doc_data)

# Instantiate the custom embeddings
embeddings = CustomEmbeddings()

# Create the vector store using the custom embedding function
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate responses
retriever = vectorstore.as_retriever()

from pprint import pprint

template = """Use the provided pieces of context from the Visualyse User Guide to help answer the question at the end.
If you don't know the answer, just say that you don't know.

CONTEXT:

```{context}```

QUESTION: {question}

HELPFUL ANSWER:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Use the callable LLM
llm_callable = CallableLLM(client)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def enter_question():
    print("About to invoke the RAG chain...")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm_callable  # Use the callable LLM here
        | StrOutputParser()
    )

    question = input("Enter your prompt: ")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("Finished invoking the RAG chain.")

while True:
    enter_question()

vectorstore.delete_collection()
