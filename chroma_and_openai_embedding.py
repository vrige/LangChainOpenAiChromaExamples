import chromadb
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

OPENAI_API_KEY = "..."

# Load documents from a web page
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
#document_text = " ".join([doc.page_content for doc in docs])

# Create an embedding for the page content using Open AI
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


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

