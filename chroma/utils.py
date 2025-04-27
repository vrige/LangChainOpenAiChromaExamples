from typing import Dict, List, Any

import chromadb
import asyncio

from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.types import Collection


class ChromaUtils:
    sync_collection_list: Dict[str, Collection]
    async_collection_list: Dict[str, AsyncCollection]

    def __init__(self):
        self.sync_collection_list = {}
        self.async_collection_list = {}
        self.sync_chroma_client = chromadb.HttpClient(host='localhost', port=8000)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.__add_async_client())

    def sync_check_if_coll_exists(self, collection_name: str) -> bool:
        col_exists = True
        try:
            coll = self.sync_chroma_client.get_collection(collection_name)
        except:
            col_exists = False
        return col_exists

    async def __add_async_client(self):
        self.async_chroma_client = await chromadb.AsyncHttpClient()

    def sync_create_coll(self, collection_name: str) -> None:
        if collection_name not in self.sync_collection_list:
            self.sync_collection_list.update({collection_name: self.sync_chroma_client.get_or_create_collection(name=collection_name)})

    def sync_add_vectors(self, collection_name: str, doc_list: List[str], id_list: List[str]) -> None:
        if collection_name not in self.sync_collection_list.keys():
            self.sync_create_coll(collection_name=collection_name)
        if collection_name in self.sync_collection_list.keys():
            collection = self.sync_collection_list[collection_name]
            collection.add(
                documents=doc_list,
                ids=id_list
            )

    def sync_query_collection(self, collection_name: str, query: str) -> Dict[str, Any]:
        if collection_name not in self.sync_collection_list.keys():
            self.sync_create_coll(collection_name=collection_name)
        if collection_name in self.sync_collection_list.keys():
            collection = self.sync_collection_list[collection_name]
            results = collection.query(
                query_texts=[query],
                n_results=2
            )
            print(results)
            return results

    def sync_fetch_data(self, collection_name: str):
        collection = self.sync_chroma_client.get_collection(name=collection_name)
        return collection.get() if collection else None

    async def async_check_if_coll_exists(self, collection_name: str) -> bool:
        col_exists = True
        try:
            coll = await self.async_chroma_client.get_collection(collection_name)
        except:
            col_exists = False
        return col_exists

    async def async_create_coll(self, collection_name: str) -> None:
        if collection_name not in self.async_collection_list:
            collection = await self.async_chroma_client.get_or_create_collection(name=collection_name)
            self.async_collection_list.update({collection_name: collection})

    async def async_add_vectors(self, collection_name: str, doc_list: List[str], id_list: List[str]) -> None:
        if collection_name not in self.async_collection_list.keys():
            await self.async_create_coll(collection_name=collection_name)
        elif collection_name in self.async_collection_list.keys():
            collection = self.async_collection_list[collection_name]
            await collection.add(
                documents=doc_list,
                ids=id_list
            )

    async def async_query_collection(self, collection_name: str, query: str):
        if collection_name not in self.async_collection_list.keys():
            await self.async_create_coll(collection_name=collection_name)
        if collection_name in self.async_collection_list.keys():
            collection = self.async_collection_list[collection_name]
            results = await collection.query(
                query_texts=[query],
                n_results=2
            )
            print(results)
            return results

    async def async_fetch_data(self, collection_name: str):
        collection = await self.async_chroma_client.get_collection(name=collection_name)
        return await collection.get() if collection else None


chroma_client = ChromaUtils()


async def testing_async():
    await chroma_client.async_add_vectors(collection_name="collection_example", doc_list=["Il napoletano è una bella lingua"], id_list=["id3"])
    await chroma_client.async_query_collection(collection_name="collection_example", query="una bella lingua?")
    fetched_data = await chroma_client.async_fetch_data("collection_example")
    print(f"your data: {fetched_data}")


def testing_sync():
    chroma_client.sync_create_coll(collection_name="collection_example")
    chroma_client.sync_add_vectors(collection_name="collection_example", doc_list=["ciao, come stai?", "vorrei dirti che la pasta è buona"], id_list=["id4", "id5"])
    chroma_client.sync_query_collection(collection_name="collection_example", query="una bella lingua?")
    fetched_data = chroma_client.sync_fetch_data("collection_example")
    print(f"your data: {fetched_data}")


# docs: https://docs.trychroma.com/docs/run-chroma/client-server
if __name__ == "__main__":
   testing_sync()
   # asyncio.run(testing_async())

