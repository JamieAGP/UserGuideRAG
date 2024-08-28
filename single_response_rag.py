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
    def __init__(self, client, model="ggml-model-Q8_0.gguf"):
        self.client = client
        self.model = model

    def __call__(self, prompt, system_prompt=None):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=512,
            stream=False  # Switch off streaming temporarily to simplify debugging
        )
        # Ensure we return a string
        print("RAW RESPONSE:", response)
        return response.choices[0].text.strip()

# Load and prepare the single text file
file_path = Path(__file__).parent / 'User Guide.txt'  # Replace with your actual file name
loader = TextLoader(file_path, encoding="utf-8")
doc_data = loader.load()

# Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
splits = text_splitter.split_documents(doc_data)

# Instantiate the custom embeddings
embeddings = CustomEmbeddings()

# Create the vector store using the custom embedding function
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate responses
retriever = vectorstore.as_retriever()

# System prompt for setting context
system_prompt = "You are a technical support AI for the software package 'Visualyse Professional Version 7'. You have specialised knowledge on how to perform simulations of \a range of radiocommunication systems within the software. Use your knowledge to help questioners perform their task."

template = """Use the provided pieces of context, in triple backticks, from the Visualyse User Guide to help answer the question at the end.
If you don't know the answer, just say that you don't know. The context of the question will always be about Visualyse Professional.

CONTEXT:

```{context}```

QUESTION: {question}

HELPFUL ANSWER:"""

custom_rag_prompt = PromptTemplate.from_template(template)

# Use the callable LLM
llm_callable = CallableLLM(client)

def format_docs(docs):
    print("\033[91mSelected Context: " + "\n\n".join(doc.page_content for doc in docs) + "\033[0m")
    return "\n\n".join(doc.page_content for doc in docs)

def enter_question():
    #print("About to invoke the RAG chain...")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} # Gives context by finding relevant text witrh retriever and formatting with format_docs
        | custom_rag_prompt # Applies prompt template to context and question inputs to make a single prompt string
        | (lambda x: llm_callable(x, system_prompt))  # Calls LLM for response based on single prompt string
        | StrOutputParser() # Processes response into a string
    )

    question = input("\033[94m\nEnter your prompt: \033[0m")
    
    for chunk in rag_chain.stream(question):    
        print("\033[94m\nResponse: \033[0m", flush=True)   
        print(chunk.replace("\n", " "), flush=True)
    #print("Finished invoking the RAG chain.")

while True:
    enter_question()

vectorstore.delete_collection()
