import os
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
import ollama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings 
#from langchain.output_parsers import StrOutputParser
#from langchain.runnables import RunnablePassthrough


first_load = True

if first_load:
    embeddings = GPT4AllEmbeddings(model="mistral")
    db_path = "/home/cisco/local_gpt2/localGPT/ollama_db"

    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )

    vectorstore = Chroma(embedding_function=embeddings,
            persist_directory=db_path, client_settings=CHROMA_SETTINGS)
    retriever = vectorstore.as_retriever()



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Refer the documents and then answer the questions. Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']



def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)
while True:
    question = input("\nEnter a query: ")
    result = rag_chain(question)
    print("Answer: ",result)

