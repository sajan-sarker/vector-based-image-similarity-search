# VectorDB based Face Similarity Search — Local Images + Pinecone

This project keeps the face-processing core (detection → alignment → embedding) from a broader 1:1 face-verification reference pipeline: instead of deciding whether two specific images match, it indexes a local gallery of face images by their embeddings and, given a new query image, returns the most similar indexed faces with a similarity score and metadata for each.

> **Scope:** every image here is a static file that already exists on local disk.

---

## Dataset

- 23 sample images 20 from [Kaggle Human Faces Dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset) and 3 from google.
- For searching four existing images (from 23 sample) and one new images was used.

## Architecture

1. **Face Detection:** Identifies the most prominent face in a local image.
2. **Alignment:** Performs a 5-point landmark similarity transform.
3. **Embedding:** Generates a 512-dimensional feature vector.
4. **Vector Database:** L2-normalized vectors are stored and queried in Pinecone using cosine similarity.

## Scope

The project focuses on generating embeddings for the most prominent face in an image, estimating the head pose angle (yaw), and indexing these features. It handles 1-to-N matching by finding the top-K most similar faces to a query image.

## Models & Configuration Summary

- **Model Pack:** InsightFace `buffalo_l`
- **Detection Model:** RetinaFace-10GF
- **Recognition Model:** ArcFace-trained ResNet50@WebFace600K
- **Execution Provider:** CPUExecutionProvider
- **Vector Dimension:** 512
- **Distance Metric:** Cosine Similarity

## Requirements

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

Notes:

- **Python 3.11+** (the Pinecone SDK requires 3.10+; this matches the original pipeline's requirement).
- On a headless server, `opencv-python-headless` is a drop-in replacement for `opencv-python` if you hit display-library issues.
- `insightface` downloads the `buffalo_l` model pack (a few hundred MB) to `~/.insightface/models/` on first use — the first run will be slower than subsequent ones.

---

## Environment & configuration

`.env.example` (copy to `.env` and fill in):

```bash
cp .env.example .env
```

```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=face-similarity-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Pinecone Setup

1. Create a free account at [pinecone.io](https://www.pinecone.io/) and generate an API key from the console.
2. Copy `.env.example` to `.env` and set `PINECONE_API_KEY` (adjust `PINECONE_INDEX_NAME` / `PINECONE_CLOUD` / `PINECONE_REGION` if you want something other than the defaults).
3. You don't need to manually create the index — `pinecone_client.get_index()` creates it automatically on first run if it doesn't exist yet, configured as:
   - `dimension=512` — must match the recognition model's output size
   - `metric="cosine"`
   - a serverless spec using your configured `cloud` / `region`

If you already have a Pinecone index you want to reuse, make sure its dimension and metric match the above — otherwise upserts will fail or similarity scores will be meaningless.

## Metadata Schema

When embeddings are upserted into Pinecone, the following metadata is attached to each vector:

- `image_name`: Original filename
- `local_path`: Absolute local file path
- `embedding_model`: The model used (e.g., `buffalo_l`)
- `angle`: Facing direction (`straight`, `left`, `right`)
- `indexed_at`: UTC timestamp of index insertion

## Running the pipeline (notepad only)

Currently, the pipeline is executed via the `Experiment.ipynb` Jupyter Notebook.

1. Make sure your virtual environment is active and dependencies are installed.
2. Verify `.env` is populated with your Pinecone API key.
3. Run the cells in `Experiment.ipynb` to index the dataset and test similarity queries.

## Summary & Discussion

This project demonstrates an efficient and privacy-conscious approach to reverse image search for faces. By computing InsightFace embeddings locally and storing only high-dimensional vectors and lightweight metadata in Pinecone, it achieves fast, scalable similarity search without risking raw image exposure.
