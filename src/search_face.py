"""
Query script: local query image -> embedding -> Pinecone similarity search.
"""
from .face_embedding import generate_face_embedding
from .pinecone_client import get_index


def search(query_image_path: str, top_k: int = 3):
    """ perform similarity search for a local query image against the indexed embeddings in Pinecone. """
    index = get_index()
    angle, query_embedding = generate_face_embedding(query_image_path)

    response = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True,
    )

    return angle, response