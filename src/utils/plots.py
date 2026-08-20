import random
import cv2
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from src.face_embedding import _get_app, _normalize

def display_random_images(face_images_dir, num_images=5):
    """Display random images in a single row."""
    image_files = list(face_images_dir.glob("*"))

    if len(image_files) < num_images:
        print(f"Not enough images to display. Found {len(image_files)} images.")
        return

    random_images = random.sample(image_files, num_images)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for ax, img_path in zip(axes, random_images):
        img = cv2.imread(str(img_path))

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")
        else:
            ax.set_title("Failed to load")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# Fixed ArcFace 112x112 landmark reference template
# It is the standard target position used for ArcFace alignment.
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)

def visualize_face_pipeline(path: Path):
    """
    Visualize the InsightFace pipeline:

        Original
            ↓
        Face Detection + Landmarks
            ↓
        Face Alignment
            ↓
        Recognition Input
    """
    # Read image
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # InsightFace detection
    # InsightFace automatically detects:
    # - face bounding box
    # - 5 facial landmarks
    # - face embedding
    faces = _get_app().get(image)

    if len(faces) == 0:
        raise ValueError(f"No face detected: {path}")

    # Select largest face
    if len(faces) > 1:
        faces.sort(
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            ),
            reverse=True,
        )
        print(f"Warning: {path} contains {len(faces)} faces. Using the largest face.")

    face = faces[0]

    # Get automatically detected landmarks
    # These coordinates come directly from InsightFace.
    detected_landmarks = np.asarray(face.kps, dtype=np.float32)

    print("\nDetected landmarks:\n", detected_landmarks)

    # Visualize detection 
    detection_image = image.copy()
    x1, y1, x2, y2 = face.bbox.astype(int)

    # Face bounding box
    cv2.rectangle(
        detection_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )

    # Automatically detected landmarks
    for point in detected_landmarks:
        x, y = point.astype(int)
        cv2.circle(
            detection_image,
            (x, y),
            5,
            (0, 0, 255),
            -1,
        )
    detection_rgb = cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB)

 
    # Perform alignment for visualization
    # InsightFace internally performs the equivalent landmark-based alignment for recognition.
    # Here we reproduce that transformation only so we can visualize the aligned face.
 
    transform_matrix, _ = cv2.estimateAffinePartial2D(
        detected_landmarks,
        ARC_FACE_TEMPLATE,
        method=cv2.LMEDS,
    )

    if transform_matrix is None:
        raise ValueError("Could not calculate face alignment transform.")

    aligned_face = cv2.warpAffine(
        image,
        transform_matrix,
        (112, 112),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

 
    # Recognition input
    # This is the standardized face representation that is fed into the ArcFace recognition model. 
    recognition_input = aligned_face.copy()
    recognition_rgb = cv2.cvtColor(recognition_input, cv2.COLOR_BGR2RGB)
 
    # actual InsightFace embedding
    embedding = _normalize(face.embedding)


    # Display all four stages 
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Detection
    axes[1].imshow(detection_rgb)
    axes[1].set_title("Face Detection + Landmarks", fontsize=13)
    axes[1].axis("off")

    # Alignment
    axes[2].imshow(aligned_rgb)
    axes[2].set_title("Aligned Face (112×112)", fontsize=13)
    axes[2].axis("off")

    # Recognition input
    axes[3].imshow(recognition_rgb)
    axes[3].set_title("Recognition Input", fontsize=13)
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

 
    # Print embedding information
    print("\nEmbedding information:")
    print("Dimension :", embedding.shape)
    print("Dtype     :", embedding.dtype)
    print("Norm      :", np.linalg.norm(embedding))
    print(f"Embedding: {embedding[:5]}... {embedding[-5:]}")


def display_query_and_best_match(query_image_path, angle, search_result):
    """Display the query image and the highest-scoring matched image side-by-side. """
    # Get highest-scoring match
    best_match = max(search_result.matches, key=lambda match: match.score)
    best_score = best_match.score
    if best_score < 0.6:
        print(f"No Match Found!")
        return
    matched_image_path = best_match.metadata["local_path"]

    # Load query image
    query_image = cv2.imread(query_image_path)
    if query_image is None:
        raise ValueError(f"Could not load query image: {query_image_path}")

    # Load matched image
    matched_image = cv2.imread(matched_image_path)
    if matched_image is None:
        raise ValueError(f"Could not load matched image: {matched_image_path}")
    
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    matched_image = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image",fontsize=14)
    axes[0].axis("off")

    # Best matched image
    axes[1].imshow(matched_image)
    axes[1].set_title(
        f"Best Match\n"
        f"{best_match.metadata['image_name']}\n"
        f"Similarity Score: {best_score:.4f}\n"
        f"Angle: {best_match.metadata['angle']}",
        fontsize=14
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


    print("Best Match:")
    print("ID       :", best_match.id)
    print("Image    :", best_match.metadata["image_name"])
    print("Angle    :", best_match.metadata["angle"])
    print("Score    :", best_score)
    print("Path     :", matched_image_path)