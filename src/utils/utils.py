import os
from pathlib import Path


def check_face_images_directory(face_images_dir):
    if not face_images_dir.exists():
        print(f"Directory {face_images_dir} does not exist.")
    else:
        image_files = list(face_images_dir.glob("*"))
        if not image_files:
            print(f"No images found in {face_images_dir}.")
        else:
            print(f"Found {len(image_files)} images in {face_images_dir}:")
            for img in image_files:
                print(f" - {img.name}")