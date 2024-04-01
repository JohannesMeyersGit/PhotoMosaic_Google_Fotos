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

org_im = Image.open(r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240310_101847.jpg')

list_of_ims = [r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240308_095344.jpg',r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240309_153856.jpg',r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240310_061607.jpg',r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240310_101838.jpg'
               ,r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240313_064724.jpg', r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240314_170443.jpg', r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_060718.jpg',r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_060836.jpg'
               ,r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_062843.jpg',r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240319_071632.jpg', r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240320_180148.jpg', r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\20240321_181036.jpg'
               ,r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\IMG_20240322_223939.jpg']


m,n,c = np.array(org_im).shape
print('Shape of the original image:', m,n,c)  # Shape of the original image: 2268 4032 3

# set the size of the mosaic tiles
tile_size = 30 # size of the mosaic tiles --|> 50x50 pixels
N_random_neighbors = 40 # number of random nearest neighbors to choose from the KDTree
use_alpha = True # use alpha blending to blend the mosaic image with the original image
alpha = 0.9 # alpha value for blending the mosaic image with the original image to increase the color correctness of the mosaic image
precalculate_tiles = True # precalculate the rgb values of the images and downsample the images to the tile size for faster processing

add_letters = False # add letters to the original image 

# resize the original image to a multiple of the tile size
m = m - m % tile_size
n = n - n % tile_size
org_im = org_im.resize((n, m))
m,n,c = np.array(org_im).shape

print('Shape of the original image after resizing:', m,n,c)  # Shape of the original image after resizing: 2268 4032 3

# calculate the number of tiles in the x and y directions
num_tiles_x = n // tile_size
num_tiles_y = m // tile_size

print('Number of tiles in x direction:', num_tiles_x)  # Number of tiles in x direction: 20
print('Number of tiles in y direction:', num_tiles_y)  # Number of tiles in y direction: 11

print('Total number of tiles:', num_tiles_x*num_tiles_y)  # Total number of tiles: 220


org_im = np.array(org_im)


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
        image = Image.open(im_path)
        image = im.resize((1, 1)) # resize to 1x1 pixels to get the average color
        rgb_values[:, i] = np.array(image).squeeze()
        
    # save the rgb values to a file for future use

    np.save('rgb_values.npy', rgb_values)


# pick the 10 most common colors from the rgb values using k-means clustering

from sklearn.cluster import KMeans

N_colors = 1000

kmeans = KMeans(n_clusters=N_colors)

# fit the kmeans model to the rgb values
kmeans.fit(rgb_values.T)


# get the cluster centers
cluster_centers = kmeans.cluster_centers_

# get the cluster labels
cluster_labels = kmeans.labels_

# get the number of images in each cluster

cluster_counts = np.bincount(cluster_labels)

# get the N most common colors

N_most_common_colors = cluster_centers[np.argsort(cluster_counts)[-N_colors:]]
print('N most common colors:', N_most_common_colors)

# create a list of the RGB values of the N most common colors

N_most_common_colors = N_most_common_colors.astype(np.uint8)

# get the 10 colors most different colors from the N most common colors

# calculate the cosine similarity between the N most common colors and all the colors
from sklearn.metrics.pairwise import cosine_similarity

# calculate the cosine similarity between the N most common colors and all the colors

# normalize the N most common colors
N_most_common_colors_norm = N_most_common_colors / np.linalg.norm(N_most_common_colors, axis=1)[:, np.newaxis]

# normalize the rgb values
rgb_values_norm = rgb_values / np.linalg.norm(rgb_values, axis=0)
cosine_sim = cosine_similarity(N_most_common_colors_norm, rgb_values_norm.T)

# remove the 20% most similar colors
cosine_sim[cosine_sim > 0.9] = 0

# get the 10 most different colors from the N most common colors

N_random_colors = rgb_values[:, np.argsort(np.min(cosine_sim, axis=0))[::10]]


# generate letters as org image for testing purposes

letter_im = np.ones((m, n, c), dtype=np.uint8)*255 # create a blank image with the same size as the original image


# write the letters to the center of the image
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 40
thickness = 150
letter_spacing = int(n//5)

line_spacing = 1000
# Define the colors for each letter
colors = {
    'G': N_random_colors.T[0].tolist(),
    'u': N_random_colors.T[1].tolist(),
    'a': N_random_colors.T[2].tolist(),
    't': N_random_colors.T[3].tolist(),
    'e': N_random_colors.T[4].tolist(),
    '2': N_random_colors.T[5].tolist(),
    '0': N_random_colors.T[6].tolist(),
    '2': N_random_colors.T[7].tolist(),
    '4': N_random_colors.T[8].tolist()
}
if add_letters == True:
    # Colorize the letters
    cv2.putText(org_im, 'G', (n//2 - int(2.5*letter_spacing), m//2 - int(line_spacing*0.25)), font, font_scale, colors['G'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, 'u', (n//2 - int(1.5*letter_spacing), m//2 - int(line_spacing*0.25)), font, font_scale, colors['u'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, 'a', (n//2 - int(0.5*letter_spacing), m//2 - int(line_spacing*0.25)), font, font_scale, colors['a'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, 't', (n//2 + int(0.5*letter_spacing), m//2 - int(line_spacing*0.25)), font, font_scale, colors['t'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, 'e', (n//2 + int(1.2*letter_spacing), m//2 - int(line_spacing*0.25)), font, font_scale, colors['e'], thickness, cv2.LINE_AA)

    cv2.putText(org_im, '2', (n//2 - 2*letter_spacing, m//2 + int(line_spacing*0.75)), font, font_scale, colors['2'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, '0', (n//2 - 1*letter_spacing, m//2 + int(line_spacing*0.75)), font, font_scale, colors['0'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, '2', (n//2 + 0*letter_spacing, m//2 + int(line_spacing*0.75)), font, font_scale, colors['2'], thickness, cv2.LINE_AA)
    cv2.putText(org_im, '4', (n//2 + 1*letter_spacing, m//2 + int(line_spacing*0.75)), font, font_scale, colors['4'], thickness, cv2.LINE_AA)

# show image using matplotlib
plt.figure()
plt.imshow(np.array(org_im))
plt.title('Original Image')
plt.show()

# shape of the original image


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
                image = Image.open(im_path)
                image = image.resize((tile_size, tile_size))
                tile_ims[:, :, :, i] = np.array(image)
    
    # save the tile images for future use
    if precalculate_tiles == True:
        np.save(name_tile_stack, tile_ims)

# create a KDTree for fast nearest neighbor search of the rgb values
kdtree = cKDTree(rgb_values.T)

# get N nearest neighbors for each pixel in the original image and create the mosaic image

# create a blank image to store the mosaic image with the same size as the original image


# loop through each tile in the original image and get the nearest neighbor from the rgb values

# add tqdm progress bar

PBar = tqdm.tqdm(total=No_of_ims)




for k,im_path in enumerate(list_of_ims):
    org_im = Image.open(im_path)
    m,n,c = np.array(org_im).shape
    # resize the original image to a multiple of the tile size
    m = m - m % tile_size
    n = n - n % tile_size
    org_im = org_im.resize((n, m))
    m,n,c = np.array(org_im).shape

    # calculate the number of tiles in the x and y directions
    num_tiles_x = n // tile_size
    num_tiles_y = m // tile_size

    name_of_image = os.path.basename(im_path)
    # get the average rgb value of the tiles using numpy mean function and block reduce

    Avgs = block_reduce(np.array(org_im), (tile_size, tile_size, 1), np.mean)
    mosaic_im = np.zeros((m, n, c))
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
        mosaic_im = alpha*mosaic_im + (1-alpha)*np.array(org_im)

    # save the mosaic image to a file using the tile size and the original image name
    mosaic_im = Image.fromarray(mosaic_im.astype(np.uint8))
    mosaic_im.save('mosaic_'+str(tile_size)+'_'+name_of_image)



# show the mosaic image using matplotlib

plt.figure()
plt.imshow(mosaic_im)
plt.title('Mosaic Image')
plt.show()



    