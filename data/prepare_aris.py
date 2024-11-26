import os
import cv2
import h5py
import argparse
import pandas as pd
import numpy as np
from PIL import Image


def extract_coordinates_from_csv(csv_file):
    """
    从CSV文件中提取坐标信息。

    参数：
    csv_file (str): CSV文件的路径。

    返回值：
    np.ndarray: 包含所有坐标信息的数组，每个坐标以(x, y)形式存储。
    """
    df = pd.read_csv(csv_file)
    fish_coordinates = df[['x', 'y']].values
    return fish_coordinates


# Get the directory of the current Python file and set it as the working directory
current_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_folder)
print("Working directory set to:", os.getcwd())

# Argument parser
parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--data_path', type=str, default='../datasets/ARIS', help='The data path of the ARIS dataset')
args = parser.parse_args()

# Load lists for train, val, and test sets
with open("./ARIS_list/train_id.txt", "r") as f:
    train_list = f.readlines()

with open("./ARIS_list/val_id.txt", "r") as f:
    val_list = f.readlines()

with open("./ARIS_list/test_id.txt", "r") as f:
    test_list = f.readlines()

# Define a function to process images for any given list


def process_dataset(image_list, dataset_type):
    """
    Processes images in a dataset (train/val/test) by creating ground truth maps and saving them in .h5 format.

    参数：
    image_list (list): 包含图像文件名的列表。
    dataset_type (str): 数据集类型（'train'，'val' 或 'test'）。
    """
    for img_name in image_list:
        fname = img_name.strip()
        img_path = os.path.join(args.data_path, 'images', fname)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image not found: {img_path}")
            continue

        # Load image in PIL format
        Img_data_pil = Image.open(img_path).convert('RGB')

        # Initialize ground truth maps
        k = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        point_map = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)

        # Load ground truth coordinates from CSV
        mat_path = img_path.replace('images', 'annotations').replace('.jpg', '.csv')
        gt = extract_coordinates_from_csv(mat_path)

        # Mark points on ground truth maps
        for coord in gt:
            x, y = int(coord[0]), int(coord[1])
            if y < img.shape[0] and x < img.shape[1]:
                k[y, x] = 1
                cv2.circle(point_map, (x, y), 5, (0, 0, 0), -1)

        # Convert point map to uint8 format for saving
        kpoint = k.astype(np.uint8)

        # Save the image and ground truth to .h5 file
        output_path = img_path.replace('images', 'gt_detr_map').replace('.jpg', '.h5')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with h5py.File(output_path, 'w') as hf:
            hf['kpoint'] = kpoint
            hf['image'] = np.array(Img_data_pil)  # Store PIL image as an array

        print(f"{dataset_type}_part", img_path)


# Process each dataset
process_dataset(train_list, 'train')
process_dataset(val_list, 'val')
process_dataset(test_list, 'test')
