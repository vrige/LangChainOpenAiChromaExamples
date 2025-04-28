from langchain_community.document_loaders import WebBaseLoader
from openai_utils import OpenAI

OPENAI_API_KEY = "..."

# Load documents from a web page
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
document_text = " ".join([doc.page_content for doc in docs])

# Create an embedding for the page content using Open AI
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

print(get_embedding(document_text))