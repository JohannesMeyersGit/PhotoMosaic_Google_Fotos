import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.http import MediaFileUpload
import os
import pickle
import requests
# add progress bar to download
from tqdm import tqdm


class GooglePhotosApi:
    def __init__(self,
                 api_name = 'photoslibrary',
                 client_secret_file= r'./client_secret.json',
                 api_version = 'v1',
                 scopes = ['https://www.googleapis.com/auth/photoslibrary']):
        '''
        Args:
            client_secret_file: string, location where the requested credentials are saved
            api_version: string, the version of the service
            api_name: string, name of the api e.g."docs","photoslibrary",...
            api_version: version of the api

        Return:
            creditial object or none
        
        Information:
        
        To enable google photos api, you need to create a project in the google cloud console and enable the google photos api.
        After enabling the api, you need to create credentials for the api.
        
        The credentials are saved in a json file, which is used to authenticate the user.
        The credentials are saved in a pickle file, so that the user does not have to authenticate every time.
        If the credentials are expired, the user has to authenticate again.
        
        For more details see: https://developers.google.com/photos/library/guides/get-started and the nice tutorial from https://github.com/polzerdo55862/google-photos-api/blob/main/Google_API.ipynb
        '''

        self.api_name = api_name
        self.client_secret_file = client_secret_file
        self.api_version = api_version
        self.scopes = scopes
        self.cred_pickle_file = f'./token_{self.api_name}_{self.api_version}.pickle'

        self.cred = None

    def run_local_server(self):
        # is checking if there is already a pickle file with relevant credentials
        if os.path.exists(self.cred_pickle_file):
            with open(self.cred_pickle_file, 'rb') as token:
                self.cred = pickle.load(token)

        # if there is no pickle file with stored credentials, create one using google_auth_oauthlib.flow
        if not self.cred or not self.cred.valid:
            if self.cred and self.cred.expired and self.cred.refresh_token:
                self.cred.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, self.scopes)
                self.cred = flow.run_local_server()

            with open(self.cred_pickle_file, 'wb') as token:
                pickle.dump(self.cred, token)
        
        return self.cred
    

class AccessPhotos():
    
    def __init__(self, credentials):
        self.credentials = credentials
        self.service = build('photoslibrary', 'v1', credentials=credentials, static_discovery=False)
        
    def get_albums(self):
        results = self.service.albums().list().execute()
        albums = results.get('albums', [])
        return albums
    
    
    def get_media_items(self):
        results = self.service.mediaItems().list().execute()
        media_items = results.get('mediaItems', [])
        return media_items
    
    def get_shared_album(self, share_token):
        album = self.service.sharedAlbums().get(shareToken=share_token).execute()
        return album
    
    def get_shared_album_media_items(self, share_token):
        results = self.service.sharedAlbums().listMediaItems(shareToken=share_token).execute()
        media_items = results.get('mediaItems', [])
        return media_items
    
    def upload_media_item(self, file_path, album_id):
        media = MediaFileUpload(file_path)
        request_body = {
            'albumId': album_id,
            'newMediaItems': [{'description': 'test_description', 'simpleMediaItem': {'uploadToken': media}}]
        }
        response = self.service.mediaItems().batchCreate(body=request_body).execute()
        return response
    
    def upload_media_item_simple(self, file_path, album_id):
        media = MediaFileUpload(file_path)
        request_body = {
            'albumId': album_id,
            'newMediaItems': [{'description': 'test_description', 'simpleMediaItem': {'uploadToken': media}}]
        }
        response = self.service.mediaItems().batchCreate(body=request_body).execute()
        return response
    
    def create_album(self, title):
        request_body = {
            'album': {'title': title}
        }
        response = self.service.albums().create(body=request_body).execute()
        return response
    
    def add_media_item_to_album(self, media_item_id, album_id):
        request_body = {
            'mediaItemIds': [media_item_id]
        }
        response = self

    def get_media_items_by_album(self, album_id):
        results = self.service.mediaItems().search(body={'albumId': album_id}).execute()
        media_items = results.get('mediaItems', [])
        return media_items
    
    def get_album_id_by_title(self, title):
        albums = self.get_albums()
        for album in albums:
            if album['title'] == title:
                return album['id']
        return None
    
    def get_all_media_items_by_album_id(self, album_id):
        media_items = []
        results = self.service.mediaItems().search(body={'albumId': album_id}).execute()
        media_items.extend(results.get('mediaItems', []))
        while 'nextPageToken' in results:
            results = self.service.mediaItems().search(body={'albumId': album_id, 'pageToken': results['nextPageToken']}).execute()
            media_items.extend(results.get('mediaItems', []))
        return media_items
    
    def download_all_media_items_by_album_id(self, album_id, target_directory, videos=True, photos=True):
        media_items = self.get_all_media_items_by_album_id(album_id)
        folder_images = os.path.join(target_directory, 'images')
        folder_videos = os.path.join(target_directory, 'videos')
        if not os.path.exists(folder_images):
            os.makedirs(folder_images)
        if not os.path.exists(folder_videos):
            os.makedirs(folder_videos)
            
        
        # go through all media items and download them and show progress bar
        for media_item in tqdm(media_items):
            if media_item['mimeType'] == 'image/jpeg' and photos:
                # check if the media item already exists in the folder and skip if it does exist to save api calls
                if os.path.exists(os.path.join(folder_images, media_item['filename'])):
                    continue
                response = requests.get(media_item['baseUrl'] + '=d')
                with open(os.path.join(folder_images, media_item['filename']), 'wb') as f:
                    f.write(response.content)
            elif media_item['mimeType'] == 'video/mp4' and videos:
                # check if the media item already exists in the folder and skip if it does exist to save api calls
                if os.path.exists(os.path.join(folder_videos, media_item['filename'])):
                    continue
                response = requests.get(media_item['baseUrl'] + '=dv')
                with open(os.path.join(folder_videos, media_item['filename']), 'wb') as f:
                    f.write(response.content)
        
        return None

    
    
    
    


if __name__ == '__main__':

    # initialize photos api and create service
    google_photos_api = GooglePhotosApi()
    creds = google_photos_api.run_local_server()

    # create service
    service = AccessPhotos(creds)

    # get albums
    albums = service.get_albums()

    for album in albums:
        print(album['title'])


    # get media items for album Guate 2024

    album_id = service.get_album_id_by_title('Guate 2024')
    
    # download all media items from album Guate 2024
    service.download_all_media_items_by_album_id(album_id, 'D:\PythonCode\Photo_Mosaic_Google_Fotos\cache', videos=True, photos=True)

    #media_items = service.get_all_media_items_by_album_id(album_id)

    # number of all media items in album Guate 2024
    #print(len(media_items))
        
    # number of .jpg files in album Guate 2024
    #print(len([media_item for media_item in media_items if media_item['mimeType'] == 'image/jpeg']))
    
    # number of .mp4 files in album Guate 2024
    #print(len([media_item for media_item in media_items if media_item['mimeType'] == 'video/mp4']))
    
    




