# read all folders in C:\Users\ldeir\OneDrive\Desktop\ai\datasets
import os
import glob
import cv2
import numpy as np

folder_path = "C:/Users/ldeir/OneDrive/Desktop/ai/datasets"
# each folder in 'folder' contains a test, train and valid folder
dataset_folders = os.listdir(folder_path)

# dictionary to hold the number of images in each folder
dataset_images = {}

# add up the number of images in each folder
for dataset in dataset_folders:
    # get the path to the folder
    path = os.path.join(folder_path, dataset)

    # each path has a test, train and valid folder
    test_path = os.path.join(path, "test")
    train_path = os.path.join(path, "train")
    valid_path = os.path.join(path, "valid")

    # each of the above has a folder for images and labels
    test_images_path = os.path.join(test_path, "images")
    train_images_path = os.path.join(train_path, "images")
    valid_images_path = os.path.join(valid_path, "images")

    # get the number of images in each folder
    test_images = len(os.listdir(test_images_path))
    train_images = len(os.listdir(train_images_path))
    valid_images = len(os.listdir(valid_images_path))

    # add the number of images to the dictionary
    dataset_images[dataset] = {
        "test": test_images,
        "train": train_images,
        "valid": valid_images,
        "total": test_images + train_images + valid_images
    }

    # print the number of images in each folder
    print(f"Dataset: {dataset}")
    print(f"Test images: {test_images}")
    print(f"Train images: {train_images}")
    print(f"Valid images: {valid_images}")

# print the total number of images in each folder starting from the largest
sorted_dataset_images = sorted(dataset_images.items(), key=lambda x: x[1]['total'], reverse=True)
print("\nTotal images in each dataset folder:")
for dataset, images in sorted_dataset_images:
    print(f"{dataset}: {images['total']} images")