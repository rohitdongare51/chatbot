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



DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    # ".pdf": PDFMinerLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

source_dir = "/home/cisco/local_gpt2/localGPT/SOURCE_DOCUMENTS/test/"

def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
     #Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None

def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    docs = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)
                #load_single_document()

    for path in paths:

        #print(path)
        docs.append(load_single_document(path))
    return docs

def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs

print(source_dir)
documents = load_documents(source_dir)
print("Loaded docs are",documents)
text_documents, python_documents = split_documents(documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
texts = text_splitter.split_documents(text_documents)
texts.extend(python_splitter.split_documents(python_documents))
embeddings = GPT4AllEmbeddings(model="mistral")
print("creating Embeddings")
db_path = "/home/cisco/local_gpt2/localGPT/ollama_db"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

vectorstore = Chroma.from_documents(documents=texts,embedding=embeddings,persist_directory=db_path, client_settings=CHROMA_SETTINGS)
retriever = vectorstore.as_retriever()



#def format_docs(docs):
#    return "\n\n".join(doc.page_content for doc in docs)
#
## Define the Ollama LLM function
#def ollama_llm(question, context):
#    formatted_prompt = f"Question: {question}\n\nContext: {context}"
#    response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': formatted_prompt}])
#    return response['message']['content']
#
#
#
#def rag_chain(question):
#    retrieved_docs = retriever.invoke(question)
#    formatted_context = format_docs(retrieved_docs)
#    print("Entering Ollama llm")
#    return ollama_llm(question, formatted_context)
#
#result = rag_chain(" Write python code to get device records from FMC")
#print(result)

