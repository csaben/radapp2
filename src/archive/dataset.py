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

    # Create image from average pixel data for saving
    # image = Image.fromarray(average_pixel_data)
    # image = image.convert('L')
    # image.save('../output/average_image.jpg')
    # os.system("scp arelius@192.168.2.67:/workspace/radapp2/output/average_image.png 'C:/Users/Clark Saben'")
    #scp not working, need to figure that out eventually
    
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
    """
    ds_helper() ; given file directory, finds the S# dir, for:each s_type in each S# convert dcm to avg img, add to
              an array to return at end of function (effectively takes tumor or control and returns set of images)

    expected ../input/DCM/DCM/Control or ../input/DCM/DCM/Tumor

    objective step into P#/S# convert and make an average image out of all the S type directories
    per P# and add the image to the numpy array that should be of the form np.array(img, label)
    """
    dcm_dirs = find_dcm_dirs(path)
    print(dcm_dirs)
    image = avg_img(dcm_dirs[0])


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
