import folium
from PIL import Image
from PIL.ExifTags import TAGS

# Function to extract GPS coordinates from photo metadata
def get_gps_coordinates(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    gps_info = {}
    if exif_data is not None:
        for tag, value in exif_data.items():
            if tag in TAGS and TAGS[tag] == 'GPSInfo':
                for key in value.keys():
                    sub_tag = TAGS.get(key, key)
                    gps_info[sub_tag] = value[key]
    return gps_info

# Example usage
image_path = r'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache\images\IMG_20240312_062531.jpg'
gps_coordinates = get_gps_coordinates(image_path)
print(gps_coordinates)

# Plotting the map
map = folium.Map(location=[gps_coordinates['GPSLatitude'], gps_coordinates['GPSLongitude']], zoom_start=15)
folium.Marker([gps_coordinates['GPSLatitude'], gps_coordinates['GPSLongitude']]).add_to(map)
map.save('/path/to/map.html')
