"""
    local face images -> embeddings -> Pinecone.
"""

import os
from datetime import datetime, timezone
from glob import glob

from . import config
from .face_embedding import generate_face_embedding, MODEL_PACK
from .pinecone_client import get_index

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

def vector_id_for(image_path: str) -> str:
    """Deterministic ID from the filename, so re-running this script on the
    same images upserts rather than creating duplicates."""
    stem = os.path.splitext(os.path.basename(image_path))[0]  # strip directory and extension
    return f"face::{stem}"

def index_directory(directory: str) -> int:
    """Index all supported images in a directory."""
    index = get_index()

    # Find all supported images in the directory (non-recursive)
    image_paths = sorted(
        p for p in glob(os.path.join(directory, "*"))
        if p.lower().endswith(SUPPORTED_EXTENSIONS)
    )

    if not image_paths:
        print(f"No supported images found in {directory}")
        return 0

    vectors = []
    for path in image_paths:
        try:
            embedding = generate_face_embedding(path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipped {path}: {e}")
            continue

        vectors.append({
            "id": vector_id_for(path),
            "values": embedding.tolist(),
            "metadata": {
                "image_name": os.path.basename(path),
                "local_path": os.path.abspath(path),
                "embedding_model": MODEL_PACK,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
        })
        print(f"Embedded {os.path.basename(path)}  ({len(embedding)}-dim)")

    if vectors:
        index.upsert(vectors=vectors)
        print(f"\nUpserted {len(vectors)} face embedding(s) into Pinecone.")