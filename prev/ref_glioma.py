#GPU not enabled via oasis labs

#dependencies 

import sys
from functools import partial
import tensorflow as tf
from tensorflow import keras
import numpy as np
import imageio as iio

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import imutils
import os
import matplotlib.pyplot as plt
# import png #need to update my build for the docker to have this
from PIL import Image #may need to add this to docker file

model = keras.models.load_model(os.path.join( os.getcwd(), "model_v1.h5"))

# Read cmdline parameters. We skip error checking in this simplified example.
input_path = sys.argv[1]
output_path = sys.argv[2]

# output_path = '/home/arelius/workspace/dockertest/test_workdir/data/out/prediction.txt'#temporary
#                /home/arelius/workspace/radapp-production/test_workdir/data/out/prediction.txt
root_dir = os.getcwd()
model = keras.models.load_model(os.path.join(root_dir, "model_v1.h5"))
#the built container will handle the unzipping ergo no need for zip file in directory when done

# input_path = os.path.join(root_dir, 'test_workdir/data/in')
input_path = 'test_workdir/data/in' #mounted so we can change after talking with albert about parcel compatibility
# == REPLACE in the terminal this is effectively the location of the P# file of a given patient from parcel

# gets all the subfolders in a given path
def get_all_subfolder_names(path):
    subdirs = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdirs.append(os.path.join(root, dir))
    return subdirs

def get_subfolder_at_depth(depth):
    offset = 1
    depth += offset
    folder_names = []
    for subfolder in get_all_subfolder_names(input_path):
        if(subfolder.count('/')==depth):
            folder_names.append(subfolder)      
    return(folder_names)

dicom_screenings = get_subfolder_at_depth(3)
dicom_screenings_to_merge = get_subfolder_at_depth(4)

#TEMPORARY assumes the depth of the screenings is constant; likely wont work IRL without data massaging
def set_array():
    listToStr = ''.join(map(str, dicom_screenings_to_merge))
    array = []
    for file in os.listdir(listToStr):
        if file.endswith(".dcm"):
            file = f"{listToStr}/{file}"
            array.append(file)
        else:
            continue
    return array

test = set_array()
print(test)

def avg_img(pxl_array):
    array = pxl_array
    length = len(array)
    sum_img = np.zeros((pydicom.dcmread(array[0])).pixel_array.astype(float).shape)
    for dicom in range(len(array)):
        # print(array[dicom])
        pyread = pydicom.dcmread(array[dicom])
        updated_img = pyread.pixel_array.astype(float)
        sum_img = np.add(sum_img, updated_img)
    avg_img = sum_img/length 

    # print(avg_img) #appears to work


    image_2d_scaled = (np.maximum(avg_img,0) / avg_img.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file try again later, for now lets just use jpeg
    # with open(destination, 'wb') as png_file:
    #     w = png.Writer(shape[1], shape[0], greyscale=True)
    #     w.write(png_file, image_2d_scaled)

    operator = 'avg_img.jpeg'
    im = Image.fromarray(image_2d_scaled)
    im.save(operator)
    return operator

avg_img = avg_img(test)

'''Following Visualization code should be scraped after initial setup'''

#cropping and subsequently visualizing

def crop_brain_contour(image, plot=False):
    
    # Convert the image to grayscale, and blur it slightly
    # gray = cv2.cvtColor(np.float32(imgUMat), cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.tick_params(axis='both', which='both',top=False, bottom=False, left=False, right=False,labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        plt.title('Cropped Image')
        plt.show()
    
    return new_image



image_loc = avg_img

def get_image_from_dicom():
    image = cv2.imread(image_loc)
    cv_image = crop_brain_contour(image, True)
    img = cv2.resize(cv_image,(240,240))     # resize image to match model's expected sizing
    img = img.reshape(1,240,240,3)  
    return(img)


#Prediction
img = get_image_from_dicom()
x = model.predict(img)

#Output for model prediction [still missing Confidence percentage]

with open(output_path, 'w') as f:
    f.write(f"test succeeded: {x[0][0]}")
    #where x stores the prediction value in x[0][0] and 1.0 is tumor and 0.0 is no tumor

'''
commands for building and using docker image

here is the universal setup and run script:

sudo docker build -t arelius/glioma-demo:TAG .

sudo docker push arelius/glioma-demo:TAG

sudo docker run -it --mount src="/home/arelius/workspace/radapp-production",target=/radapp-demo,type=bind arelius/glioma-demo:prod2 REPLACE ./test_workdir/data/out/prediction.txt
'''
