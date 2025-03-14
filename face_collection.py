import os
import cv2
import numpy as np

# Function to load images and labels from folders
def load_images_from_folders(data_dir):
    faces_data = []  # List to store image data
    labels = []      # List to store labels
    label_dict = {}  # Dictionary to map folder names to numeric labels
    current_label = 0  # Start labeling from 0

    # Iterate through each folder (person) in the data directory
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):  # Check if it's a directory
            label_dict[person_name] = current_label  # Assign a numeric label to the folder
            # Iterate through each image in the folder
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
                if image is not None:
                    resized_image = cv2.resize(image, (50, 50))  # Resize image to 50x50
                    faces_data.append(resized_image.flatten())  # Flatten image to 1D array
                    labels.append(current_label)  # Assign the label to the image
            current_label += 1  # Increment label for the next folder

    return np.array(faces_data), np.array(labels), label_dict

# Load images and labels
data_dir = "data"  # Path to your data folder
faces_data, labels, label_dict = load_images_from_folders(data_dir)

# Print some information
print(f"Loaded {len(faces_data)} images.")
print(f"Number of unique labels: {len(label_dict)}")
print("Label mapping:", label_dict)