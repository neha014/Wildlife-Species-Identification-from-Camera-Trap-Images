import os
import shutil
import random

# Path to your raw dataset
RAW_DATA_DIR = "raw-img"
OUTPUT_DIR = "datasets/animals"

# Split ratio
TRAIN_RATIO = 0.8

# Create train/val folders
for split in ["train", "val"]:
    split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

# Loop over each class
for class_name in os.listdir(RAW_DATA_DIR):
    class_path = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # Make class folders inside train and val
    os.makedirs(os.path.join(OUTPUT_DIR, "train", class_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", class_name), exist_ok=True)

    # Get all images and shuffle
    images = os.listdir(class_path)
    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_RATIO)

    # Move images to train/val
    for i, img_name in enumerate(images):
        src = os.path.join(class_path, img_name)
        if i < split_idx:
            dst = os.path.join(OUTPUT_DIR, "train", class_name, img_name)
        else:
            dst = os.path.join(OUTPUT_DIR, "val", class_name, img_name)
        shutil.copy(src, dst)

print("Dataset split into train/val successfully!")
