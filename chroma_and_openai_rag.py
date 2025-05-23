from flask import Flask, jsonify, request
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma.utils import ChromaUtils
from openai_utils.openai_utils import OpenaiAgent

chroma_client = ChromaUtils()
chroma_client.sync_create_coll(collection_name="your_coll_name")

app = Flask(__name__)


def store_data(url: str) -> None:

    # Step 1 - Load documents from a web page
    loader = WebBaseLoader(web_path=url)
    docs = loader.load()
    if docs:
        # Step 2 - Creating embedding and splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        documents_text = [doc.page_content for doc in documents]

        # Step 3 - Store the data
        ids = chroma_client.sync_add_vectors(
            collection_name="your_coll_name",
            doc_list=documents_text
        )


@app.route('/conversate', methods=['POST'])
def main():
    input_json = request.get_json(force=True)
    query = input_json["query"] if "query" else None

    if not query:
        return jsonify({"output":"Error: please provide a \'query\' field."})

    if "url" in input_json:
        store_data(input_json["url"])

    # Step 4 - Create Agent
    agent = OpenaiAgent(model="gpt-4o")

    # Step 5 - Retrieve
    results = chroma_client.sync_query_collection(
        collection_name="your_coll_name",
        query = query
    )
    context = "\n".join(results["documents"][0])

    # Step 6 - Generate
    output = agent.sync_call(content=query, prompt_path="prompts/agent_prompt.txt", placeholders={"context": context})

    print(f"agent prompt: {agent.prompt}")

    return jsonify({"query": query, "output":output})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)
