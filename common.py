"""Shared setup: models, MongoDB vector store, index loading."""
import os

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

import config


def setup_models(llm_model: str = config.LLM_MODEL) -> None:
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY  # also picked up by RAGAS
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    Settings.llm = OpenAI(model=llm_model, api_key=config.OPENAI_API_KEY, temperature=0)
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP


def get_client() -> MongoClient:
    return MongoClient(config.MONGODB_URI)


def get_vector_store(client: MongoClient) -> MongoDBAtlasVectorSearch:
    # Node layout in MongoDB: {"text": ..., "embedding": [...], "metadata": {...}}
    return MongoDBAtlasVectorSearch(
        mongodb_client=client,
        db_name=config.DB_NAME,
        collection_name=config.COLLECTION_NAME,
        vector_index_name=config.VECTOR_INDEX_NAME,
    )


def load_index(client: MongoClient) -> VectorStoreIndex:
    store = get_vector_store(client)
    return VectorStoreIndex.from_vector_store(store)


def new_storage_context(client: MongoClient) -> StorageContext:
    return StorageContext.from_defaults(vector_store=get_vector_store(client))
