import cv2

def add_blur_and_scale(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (11, 11), 8,8 ,cv2.BORDER_DEFAULT)

    # Scale down the image to 800x600
    scaled_image = cv2.resize(blurred_image, (600, 800))

    # Save the blurred and scaled image with prefix "blurred_"
    output_path = image_path.replace("mosaic_image_20_IMG_20240322_214157", "blurredImage_mosaic_image_20_IMG_20240322_214157")
    cv2.imwrite(output_path, scaled_image)

# Example usage
image_path = "D:\PythonCode\Photo_Mosaic_Google_Fotos\orgImage\mosaic_image_20_IMG_20240322_214157.jpg"
add_blur_and_scale(image_path)
