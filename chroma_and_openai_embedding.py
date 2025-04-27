import os

import chromadb
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load documents from a web page
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
#document_text = " ".join([doc.page_content for doc in docs])




# Creating embedding and splitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
documents_text = [doc.page_content for doc in documents]

ids = ["id" + str(i) for i in range(len(documents))]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="first_collection")
collection.add(
    documents=documents_text,
    ids=ids
)

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)

