import os
import numpy as np
import argparse

# Create npydata directory if it doesn't exist
if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

# Set dataset path
parser = argparse.ArgumentParser(description='ARIS Dataset NPY Generator')
parser.add_argument('--aris_path', type=str, default='./datasets/ARIS',
                    help='The data path of the ARIS dataset')

args = parser.parse_args()
aris_root = args.aris_path

try:
    # Paths for training, validation, and test lists
    with open(os.path.join('data', "ARIS_list/train_id.txt"), "r") as f:
        train_list = f.readlines()

    with open(os.path.join('data', "ARIS_list/val_id.txt"), "r") as f:
        val_list = f.readlines()

    with open(os.path.join('data', "ARIS_list/test_id.txt"), "r") as f:
        test_list = f.readlines()

    # Ground truth map root path
    root = os.path.join(aris_root, 'gt_detr_map/')

    if not os.path.exists(root):
        print("The ARIS dataset path is wrong. Please check your path.")
    else:
        # Generate and save the training image list
        train_img_list = [os.path.join(root, fname.strip()) for fname in train_list]

        # Generate and save the validation image list
        val_img_list = [os.path.join(root, fname.strip()) for fname in val_list]

        # Generate and save the test image list
        test_img_list = [os.path.join(root, fname.strip()) for fname in test_list]

        # Save the generated lists as .npy files
        np.save('./npydata/aris_train.npy', train_img_list)
        np.save('./npydata/aris_val.npy', val_img_list)
        np.save('./npydata/aris_test.npy', test_img_list)

        print("Generated ARIS image lists successfully:",
              len(train_img_list), len(val_img_list), len(test_img_list))
except Exception as e:
    print("Error with the ARIS dataset path or file structure. Please check your path.")
    print("Exception:", e)
