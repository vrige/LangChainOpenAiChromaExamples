from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load documents from a web page
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# OpenAI API key
OPENAI_API_KEY = "..."
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Creating a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world-class technical documentation writer."),
    ("user", "{input}")
])

# Parser for output (String-based)
output_parser = StrOutputParser()

# Creating the chain - prompt → LLM → output parser
chain = prompt | llm | output_parser

# Invoking the chain with a user question
response = chain.invoke({"input": "How can LangSmith help with testing?"})

# Print the parsed response
print(response)




