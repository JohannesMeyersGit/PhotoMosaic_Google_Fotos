import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import glob
import tqdm

# load the original image
im = Image.open('D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_060836.jpg')

# show image using matplotlib

plt.figure()
plt.imshow(im)
plt.title('Original Image')
plt.show()

# shape of the original image

m,n,c = np.array(im).shape
print('Shape of the original image:', m,n,c)

# create lookup table for the mosaic tile rgb values

# get all the images in the cache folder
all_ims = glob.glob('D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\*.jpg')
No_of_ims = len(all_ims)

# create a 3xN numpy array to store the rgb values of the images
rgb_values = np.zeros((3, No_of_ims))

# loop through all the images and get the rgb values including the tqdm progress bar
for i, im_path in enumerate(tqdm.tqdm(all_ims)):
    im = Image.open(im_path)
    im = im.resize((1, 1)) # resize to 1x1 pixels to get the average color
    rgb_values[:, i] = np.array(im).squeeze()
    
# save the rgb values to a file for future use

np.save('rgb_values.npy', rgb_values)

    
    