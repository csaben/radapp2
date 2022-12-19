#!/usr/bin/env python
import torch
import os
import numpy as np
import re
import glob
import pydicom
from PIL import Image
import matplotlib.pyplot as plt

def main():
    path = "../input/DCM/DCM/Control"
    path = "../input/DCM/DCM/Tumor"
    ds_helper(path)
    pass

def avg_img(dcm_dir):
    #grab each dcm, convert to pixel, average together
    pixel_data_list = []
    # Iterate over all .dcm files in the directory
    for file in os.listdir(dcm_dir):
        # Load .dcm file
        dcm_file = pydicom.dcmread(os.path.join(dcm_dir, file))
        # Add pixel data to list
        pixel_data_list.append(dcm_file.pixel_array)

    # Calculate average of pixel data
    average_pixel_data = np.mean(pixel_data_list, axis=0)
    return average_pixel_data

def find_dcm_dirs(path):
    dcm_dirs = []
    for root, dirs, files in os.walk(path):
        #check if root contains a capital "S#"
        pattern = r'S[0-9]/[a-zA-Z]'
        regexp = re.compile(pattern)
        if regexp.search(root):
            dcm_dirs.append(root)
            # print(glob.glob(f'{root}/*.dcm'))
    return dcm_dirs


def ds_helper(path):
    dcm_dirs = find_dcm_dirs(path)
    #now we have a list of the directories containing dcm, avg them, and create a np arr


class GliomaDataset:
    #technically img should be imgs plural(?) this will become self evident
    def __init__(self, img, label):
        self.img = img
        self.lable = label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        return np.array(img[item], label[item])



if __name__ == '__main__':
    main()
