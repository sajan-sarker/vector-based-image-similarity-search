# VectorDB based Face Similarity Search — Local Images + Pinecone

This project provides a complete pipeline for processing face images, generating embeddings, and storing/querying them in a vector database (Pinecone). It can take a gallery of local face images, correct their orientations, detect the most prominent face, estimate the head pose angle, generate a high-dimensional embedding, and perform fast similarity search.

> **Scope:** every image here is a static file that already exists on local disk.

---

## Dataset

- Sample images can be obtained from the [Kaggle Human Faces Dataset](https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset).
- A downloader script is provided at `scripts/dataset_download.py`.
- Tested with various images to find similarities between known and queried identities.

## Architecture & Pipeline

<!-- Image of the pipeline -->
<div style="text-align: center;">
    <img src="docs/System%20Diagram.png" alt="Architecture" width="50%" height="50%">
</div>

The face processing and indexing pipeline is composed of the following sequential steps:

1. **Orientation Correction:** Uses a custom ONNX model (`orientation_model_v2_0.9882.onnx`) to detect image rotation (0°, 90°, 180°, 270°) and automatically correct upside-down or sideways images.
2. **Face Detection:** Identifies the most prominent face in a local image using the RetinaFace-10GF model.
3. **Alignment & Pose Estimation:** Performs a 5-point landmark similarity transform. Also calculates the 3D head pose (yaw) to classify the face angle as `"straight"`, `"left"`, or `"right"`.
4. **Embedding Generation:** Generates a 512-dimensional feature vector using an ArcFace-trained ResNet50 model. The embedding is L2-normalized.
5. **Vector Database:** Vectors and their associated metadata are stored and queried in Pinecone using cosine similarity.

## Codebase Structure

- `src/orientation_detection.py`: Loads the ONNX model to detect and correct image rotation.
- `src/angle_detection.py`: Determines the face angle (straight/left/right) based on InsightFace's pose array (yaw, pitch, roll).
- `src/face_embedding.py`: The core wrapper around InsightFace `buffalo_l`. It handles the full flow from detection to returning a 512-dim embedding.
- `src/pinecone_client.py`: Handles the Pinecone vector database connection and initialization.
- `src/index_faces.py`: Script to batch process a directory of local images, generate embeddings, and upsert them to Pinecone.
- `src/search_face.py`: Given a query image, searches the Pinecone index for the top-K most similar faces.
- `models/`: Stores local machine learning models (e.g., the orientation ONNX model).
- `scripts/`: Utilities for dataset downloading.
- `*.ipynb`: Jupyter notebooks (`Experiment.ipynb`, `Orientation_test.ipynb`, `Models_Insight.ipynb`) containing experimentation, testing, and exploratory code.

## Models & Configuration Summary

- **Face Model Pack:** InsightFace `buffalo_l` [Official Site](https://www.insightface.ai/solutions/face-recognition-licensing)
  - **Detection Model:** RetinaFace-10GF
  - **Recognition Model:** ArcFace-trained ResNet50@WebFace600K
- **Orientation Model:** Custom ONNX classifier (`orientation_model_v2_0.9882.onnx`) [HuggingFace Link](https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection)
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

## Environment & Configuration

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

## Running the pipeline

Currently, the pipeline is executed via the `Experiment.ipynb` Jupyter Notebook.

1. Make sure your virtual environment is active and dependencies are installed.
2. Verify `.env` is populated with your Pinecone API key.
3. Run the cells in `Experiment.ipynb` to index the dataset and test similarity queries.
4. Additional notebooks like `Orientation_test.ipynb` can be used to test the image orientation correction feature independently.

## Summary & Discussion

This project demonstrates an efficient and privacy-conscious approach to reverse image search for faces. By computing InsightFace embeddings locally (incorporating image rotation correction and head pose estimation) and storing only high-dimensional vectors and lightweight metadata in Pinecone, it achieves fast, scalable similarity search without risking raw image exposure.
