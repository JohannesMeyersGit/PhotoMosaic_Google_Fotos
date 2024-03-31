import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import glob
import tqdm

from scipy.spatial import cKDTree

from skimage.measure import block_reduce

# load the original image
im = Image.open(r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_060836.jpg')
#im = Image.open(r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_060718.jpg')
#im = Image.open(r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\IMG_20240312_063618.jpg')

im = Image.open(r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\IMG_20240326_131630.jpg')


m,n,c = np.array(im).shape
print('Shape of the original image:', m,n,c)  # Shape of the original image: 2268 4032 3

# set the size of the mosaic tiles
tile_size = 100 # size of the mosaic tiles --|> 50x50 pixels
N_random_neighbors = 20 # number of random nearest neighbors to choose from the KDTree
use_alpha = False # use alpha blending to blend the mosaic image with the original image
alpha = 0.7 # alpha value for blending the mosaic image with the original image to increase the color correctness of the mosaic image
precalculate_tiles = True # precalculate the rgb values of the images and downsample the images to the tile size for faster processing

# resize the original image to a multiple of the tile size
m = m - m % tile_size
n = n - n % tile_size
im = im.resize((n, m))
m,n,c = np.array(im).shape

print('Shape of the original image after resizing:', m,n,c)  # Shape of the original image after resizing: 2268 4032 3

# calculate the number of tiles in the x and y directions
num_tiles_x = n // tile_size
num_tiles_y = m // tile_size

print('Number of tiles in x direction:', num_tiles_x)  # Number of tiles in x direction: 20
print('Number of tiles in y direction:', num_tiles_y)  # Number of tiles in y direction: 11

print('Total number of tiles:', num_tiles_x*num_tiles_y)  # Total number of tiles: 220

# show image using matplotlib

plt.figure()
plt.imshow(im)
plt.title('Original Image')
plt.show()

# shape of the original image



# create lookup table for the mosaic tile rgb values

# get all the images in the cache folder
all_ims = glob.glob('D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\*.jpg')
No_of_ims = len(all_ims)

# create a 3xN numpy array to store the rgb values of the images
rgb_values = np.zeros((3, No_of_ims))

if os.path.exists('rgb_values.npy'):
    rgb_values = np.load('rgb_values.npy')
else:
    # loop through all the images and get the rgb values including the tqdm progress bar
    for i, im_path in enumerate(tqdm.tqdm(all_ims)):
        im = Image.open(im_path)
        im = im.resize((1, 1)) # resize to 1x1 pixels to get the average color
        rgb_values[:, i] = np.array(im).squeeze()
        
    # save the rgb values to a file for future use

    np.save('rgb_values.npy', rgb_values)

# create a 3D numpy array to store the downsampled images
tile_ims = np.zeros((tile_size, tile_size, 3, No_of_ims))

# loop through all the images and downsample the images for a given tile size and store them in an npy array including the tqdm progress bar

name_tile_stack = 'tile_values_'+str(tile_size)+'_.npy'

if os.path.exists(name_tile_stack):
    tile_ims = np.load(name_tile_stack)
else:
    print('Calculating tile values for all cached images')
    # loop through all the images and get the rgb values including the tqdm progress bar
    for i, im_path in enumerate(tqdm.tqdm(all_ims)):
                im = Image.open(im_path)
                im = im.resize((tile_size, tile_size))
                tile_ims[:, :, :, i] = np.array(im)
    
    # save the tile images for future use
    if precalculate_tiles == True:
        np.save(name_tile_stack, tile_ims)

# create a KDTree for fast nearest neighbor search of the rgb values
kdtree = cKDTree(rgb_values.T)

# get N nearest neighbors for each pixel in the original image and create the mosaic image

# create a blank image to store the mosaic image with the same size as the original image
mosaic_im = np.zeros((m, n, c))

# loop through each tile in the original image and get the nearest neighbor from the rgb values

# add tqdm progress bar

PBar = tqdm.tqdm(total=num_tiles_x*num_tiles_y)


# get the average rgb value of the tiles using numpy mean function and block reduce

Avgs = block_reduce(np.array(im), (tile_size, tile_size, 1), np.mean)

# loop through each tile in the original image and get the nearest neighbor from the rgb values







for l in range(num_tiles_y):
    for j in range(num_tiles_x):
        # get the rgb values of the current tile
        tile_avg = Avgs[l,j,:]
        # get N nearest neighbors for the tile average color
        _, idx = kdtree.query(tile_avg, N_random_neighbors)
        # pick a random nearest neighbor from the N nearest neighbors
        idx = np.random.choice(idx)
        # get the nearest neighbor image
        nn_im = tile_ims[:, :, :, idx]
        # paste the nearest neighbor image to the mosaic image
        mosaic_im[l*tile_size:(l+1)*tile_size, j*tile_size:(j+1)*tile_size, :] = np.array(nn_im)
        

        
        
        PBar.update(1)
        
# blend the mosaic image with the original image using the alpha value
if use_alpha == True:
    if alpha is None or alpha > 1:
        alpha = 0.7
    mosaic_im = alpha*mosaic_im + (1-alpha)*np.array(im)

# save the mosaic image to a file using the tile size and the original image name
mosaic_im = Image.fromarray(mosaic_im.astype(np.uint8))
mosaic_im.save('mosaic_image_'+str(tile_size)+'_.jpg')



# show the mosaic image using matplotlib

plt.figure()
plt.imshow(mosaic_im)
plt.title('Mosaic Image')
plt.show()


    