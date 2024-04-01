# Photo Mosaic from Google Fotos Album
Photo mosaic generator for a given google fotos folder to generate a photo mosaics from google photo albums 

## Prerequired
- Activate API access in google cloud following https://developers.google.com/photos/library/guides/get-started and the nice tutorial from https://github.com/polzerdo55862/google-photos-api/blob/main/Google_API.ipynb 
- Get client secret json and Oauth token from google cloud  


## Usage of the mosaic photos CLI interface

1. Prepare tile folder by downloading the photos which should be the tiles of your photo mosaic to a folder e.g. tile_images using the access_google_fotos script or manually
2. Prepare images folder by adding the photos which should be transfered into a mosaic style into a second folder e.g. original_images
3. Call the photo_mosaic_cli.py using "python.exe  photo_mosaic_cli.py --tile C:\...\tile_images  --original C:\...\original_images
4. The mosaic image will appear in the original_images folder with the prefix mosaic_image_tilesize

## Additional optional parameters for mosaic photos cli interface
| Argument          | Default value | Description |
|---------------|-------|------------|
| --size_of_tile  | 20  | Size of individual tiles. Smaller tiles retain more details of the original image         |
| --N_random_neighbours       | 20  | Pick one of N random images close to a given rgb value for a tile increase this to reduce repetitions of tile images in the output image       |
| --N_random_neighbours       | 20  | Pick one of N random images close to a given rgb value for a tile increase this to reduce repetitions of tile images in the output image       |
| --alpha       | 0.9  | Value of alpha blending of the tile image and the original image. Usefull to improve color richness and visibility of the original image structure          |
| --use_alpha       | True  | Bool to enable or disable alpha blending      |
| --preserve_original_image_size       | True  | Bool to resize image to the given input image size. Important if you want to print your tile images or for ordering them as poster  |
| --save_rgb_values       | True  | Bool to enable saving of rgb values of all your tile images to speedup processing of multiple mosaic images  |
| --save_tile_stack       | True  | Bool to enable saving of precalculated tiles for given size of all your tile images to speedup processing of multiple mosaic images  |

## Example Images
Attached an example of the input and output of the script using the default parameters. The images are downscaled to 600x800 and blured to save space and preserve privacy.
!orgImage\blurredImage_test.png !orgImage\blurredImage_mosaic_image_20.png
*Blured and downscaled input image to be mosaiced ;)  * | *Output mosaic image*
