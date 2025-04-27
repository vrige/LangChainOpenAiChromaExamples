from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma.utils import ChromaUtils


def main():

    # Load documents from a web page
    loader = WebBaseLoader("https://it.wikipedia.org/wiki/FantaSanremo")
    docs = loader.load()
    #document_text = " ".join([doc.page_content for doc in docs])


    # Creating embedding and splitter
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", "", "\n\n\n\n\n\n\n"])
    documents = text_splitter.split_documents(docs)
    documents_text = [doc.page_content for doc in documents]

    ids = ["id" + str(i) for i in range(len(documents))]

    chroma_client = ChromaUtils()
    chroma_client.sync_create_coll(collection_name="first_collection")
    chroma_client.sync_add_vectors(
        collection_name="first_collection",
        doc_list=documents_text,
        id_list=ids
    )

    results = chroma_client.sync_query_collection(
        collection_name="first_collection",
        query = "Who won the first sanremo?"
    )
    print(results)


if __name__ == "__main__":
    main()
