"""
Pinecone index setup and connection handling.

Pinecone stores ONLY the 512-dim face embeddings and their metadata
(image_name, local_path, ...). The original image files always stay on
local disk and are never uploaded.
"""

from pinecone import Pinecone, ServerlessSpec

from . import config
from .face_embedding import EMBEDDING_DIMENSION


def get_index():
    if not config.PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY is not set. Copy .env.example to .env and fill it in.")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    if not pc.has_index(config.PINECONE_INDEX_NAME):
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # (512)
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )

    return pc.Index(config.PINECONE_INDEX_NAME)
