import argparse
import os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import glob
import tqdm
from skimage.measure import block_reduce


class PhotoMosaic:
    
    def __init__(self) -> None:
        self.dir_to_tile_images = None
        self.dir_to_original_images = None
        self.save_rgb_values_flag = True # saves time if True by saving the rgb values of the tiles to a file
        self.save_tile_stack_flag = True # saves time if True by saving the tile stack to a file
        self.tile_size = 50 # size of the tiles
        self.N_random_neighbors = 20  # number of random neighbors to choose from in the kdtree
        self.alpha = 0.9 # alpha value for blending the mosaic image with the original image to improve colors and structure
        self.use_alpha = True # use alpha blending
        self.preserve_original_image_size = True # preserve the original image size
        self.name_tile_stack = 'tile_stack'+str(self.tile_size)+'.npy' # name of the file to save the tile stack
        self.rgb_values_file = 'rgb_values'+str(self.tile_size)+'.npy' # name of the file to save the rgb values
        self.No_of_org_ims = 1 # number of images to create the mosaic default one image
        self.No_of_tile_ims = 1 # number of tile images to create the mosaic default one image
        self.list_of_ims = [] # list of images to create the mosaic tiles from
        self.list_of_org_ims = [] # list of images to create the mosaic images from
        self.rgb_values = None # rgb values of the tiles to create the kdtree
        self.tile_ims = None # precalculated tile images
        self.kdtree = None # kdtree for fast nearest neighbor search of the rgb values of the tiles
        self.mosaic_im = None # resulting mosaic image
        self.Avgs = None # average rgb values of the tiles of the original image
        self.num_tiles_x = None # number of tiles in the x direction
        self.num_tiles_y = None # number of tiles in the y direction
        
    def prepare(self):
        self.load_tile_images()
        self.load_original_images()
        self.load_tile_stack()
        self.load_rgb_values()
        self.create_kdtree()
        
    def load_rgb_values(self):
        try:
            self.rgb_values = np.load(self.rgb_values_file)
        except:
            self.calculate_rgb_values()

    def save_rgb_values(self):
        np.save(self.rgb_values_file, self.rgb_values)
        
    def save_tile_stack(self):
        np.save(self.name_tile_stack, self.tile_ims)
    
    def load_tile_stack(self):
        try:
            self.tile_ims = np.load(self.name_tile_stack)
        except:
            self.create_tile_stack()
            
    
    def load_tile_images(self):
        # search for images in the directory looking for jpg and png files
        self.list_of_ims = glob.glob(os.path.join(self.dir_to_tile_images, '*.[jJ][pP][gG]'))
        self.No_of_tile_ims = len(self.list_of_ims)
        
        Exception = ValueError('No images found in the directory')
        assert self.No_of_tile_ims > 0, Exception
    
    def load_original_images(self):
        # search for images in the directory looking for jpg and png files
        self.list_of_org_ims = glob.glob(os.path.join(self.dir_to_original_images, '*.[jJ][pP][gG]'))
        self.No_of_org_ims = len(self.list_of_org_ims)
        
        Exception = ValueError('No images found in the directory')
        assert self.No_of_org_ims > 0, Exception
    
    def create_tile_stack(self):
        """
        Method to create the tile stack from the tile images in the tile image folder
        """
        self.tile_ims = np.zeros((self.tile_size, self.tile_size, 3, self.No_of_tile_ims))
        print('Number of tile images:', self.No_of_tile_ims)
        print('Calculating tile images for all cached images')
        for i, im_path in enumerate(tqdm.tqdm(self.list_of_ims)):
            image = Image.open(im_path)
            image = image.resize((self.tile_size, self.tile_size))
            self.tile_ims[:, :, :, i] = np.array(image)
        if self.save_tile_stack_flag:
            self.save_tile_stack()
    
    def calculate_rgb_values(self):
        """
        Method to calculate the average rgb values of each tile image in the tile image folder and store them
        in the rgb_values attribute of the class
        """
        self.rgb_values = np.zeros((3, self.No_of_tile_ims))
        print('Calculating rgb values for all cached images')
        for i, im_path in enumerate(tqdm.tqdm( self.list_of_ims)):
            image = Image.open(im_path)
            image = image.resize((1, 1)) # resize to 1x1 pixels to get the average color
            self.rgb_values[:, i] = np.array(image).squeeze()

        self.rgb_values = np.array(self.rgb_values)
        if self.save_rgb_values_flag:
            self.save_rgb_values()
    
    def create_kdtree(self):
        self.kdtree = cKDTree(self.rgb_values.T)
        
    
    def create_mosaic(self, image_path):
        """
        Method to create a mosaic image from an original image
        """
        org_im = Image.open(image_path)
        m,n,c = np.array(org_im).shape
        # resize the original image to a multiple of the tile size
        m = m - m % self.tile_size
        n = n - n % self.tile_size
        org_im = org_im.resize((n, m))
        self.num_tiles_x = n // self.tile_size
        self.num_tiles_y = m // self.tile_size
        Avgs = block_reduce(np.array(org_im), (self.tile_size, self.tile_size, 1), np.mean)
        mosaic_im = np.zeros((m, n, c))
        for l in range(self.num_tiles_y):
            for j in range(self.num_tiles_x):
                # get the rgb values of the current tile
                tile_avg = Avgs[l,j,:]
                # get N nearest neighbors for the tile average color
                _, idx = self.kdtree.query(tile_avg, self.N_random_neighbors)
                # pick a random nearest neighbor from the N nearest neighbors
                idx = np.random.choice(idx)
                # get the nearest neighbor image
                nn_im = self.tile_ims[:, :, :, idx]
                # paste the nearest neighbor image to the mosaic image
                mosaic_im[l*self.tile_size:(l+1)*self.tile_size, j*self.tile_size:(j+1)*self.tile_size, :] = np.array(nn_im)

        if self.use_alpha:
            if self.alpha is None or self.alpha > 1:
                self.alpha = 0.9 # default alpha value
            mosaic_im = self.alpha*mosaic_im + (1-self.alpha)*np.array(org_im)
        # from array to image
        mosaic_image = Image.fromarray(mosaic_im.astype(np.uint8))
        if self.preserve_original_image_size:
            mosaic_image = mosaic_image.resize((n, m))
            
        return mosaic_image

    def create_mosaic_images(self):
        # add a progress bar to show the progress of the mosaic image creation
        PBar = tqdm.tqdm(total=self.No_of_org_ims)
        print('Creating mosaic images')
        for image_path in self.list_of_org_ims:
            mosaic_image = self.create_mosaic(image_path)
            mosaic_image_name = 'mosaic_image_'+str(self.tile_size)+'_'+os.path.basename(image_path)
            # save the mosaic image to the original image folder 
            save_path = os.path.join(self.dir_to_original_images, mosaic_image_name)
            mosaic_image.save(save_path)
            PBar.update(1)
        PBar.close()
    
    def set_args(self, args):
        self.dir_to_tile_images = args.tile_images_dir
        self.dir_to_original_images = args.original_images_dir
        # set the optional arguments if they are provided
        
        if args.size_of_tile is not None: 
            self.tile_size = args.size_of_tile
        if args.N_random_neighbors is not None:
            self.N_random_neighbors = args.N_random_neighbors
        if args.alpha is not None:
            self.alpha = args.alpha
        if args.use_alpha is not None:
            self.use_alpha = args.use_alpha
        if args.preserve_original_image_size is not None:
            self.preserve_original_image_size = args.preserve_original_image_size
        if args.save_rgb_values is not None:
            self.save_rgb_values_flag = args.save_rgb_values
        if args.save_tile_stack is not None:
            self.save_tile_stack_flage = args.save_tile_stack

    

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create a photo mosaic from a set of tile images')
    parser.add_argument('--tile_images_dir', type=str, help='Directory to the tile images', required=True)
    parser.add_argument('--original_images_dir', type=str, help='Directory to the original images', required=True)
    parser.add_argument('--size_of_tile', type=int, help='Size of the tiles. Default is 20.', required=False)
    parser.add_argument('--N_random_neighbors', type=int, help='Number of random neighbors to choose from in the kdtree. Default is 20', required=False)
    parser.add_argument('--alpha', type=float, help='Alpha value for blending the mosaic image with the original image. Default is 0.9',required=False)
    parser.add_argument('--use_alpha', type=bool, help='Use alpha blending. Default is True',required=False)
    parser.add_argument('--preserve_original_image_size', type=bool, help='Preserve the original image size. Default is True',required=False)
    parser.add_argument('--save_rgb_values', type=bool, help='Save the rgb values of the tiles to a file to speed up the process. Default is True',required=False)
    parser.add_argument('--save_tile_stack', type=bool, help='Save the tile stack to a file to speed up the process. Default is True',required=False)
    args = parser.parse_args()
    
    mosaic = PhotoMosaic()
    mosaic.set_args(args)
    mosaic.dir_to_tile_images = 'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images'
    mosaic.dir_to_original_images = 'D:\PythonCode\Photo_Mosaic_Google_Fotos\orgImage'
    mosaic.prepare()
    mosaic.create_mosaic_images()
    
    
    
    
    
            
            
    
        
            
        
        
