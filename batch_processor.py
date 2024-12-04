from googleapiclient.discovery import build
from google.oauth2 import service_account
import sqlite3
import numpy as np

import supabase.client
import data_loader_spotify.spotify_download_script as download_module
import feature_extractor_module
import supabase
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "recommendationsys-0-e31dca12bea6.json"
PARENT_FOLDER_ID = "1Z7_77J9oJvZ8ZoBl2EU3Yx3AhZ6dmcgK"
PROJECT_URL = "https://qbmoyulmzltkzvtqslnl.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibW95dWxtemx0a3p2dHFzbG5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI0MzI3MDMsImV4cCI6MjA0ODAwODcwM30.Kg0APL06JN3Wa4Zd7J_uDM3nOoEclpcKYOA71QYN2n8"

supabase_client = supabase.create_client(PROJECT_URL, API_KEY)
'''
Note: Currently Supabase Table: Data stores 69993 samples only.
You can get the entire database in https://dbhub.io/aniketgope00/spotify_db.sqlite3
'''

def authenticate():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def create_buckets(total_samples, bucket_size):
  num_buckets = total_samples // bucket_size
  buckets = []
  for i in range(num_buckets):
    start = i * bucket_size
    end = (i + 1) * bucket_size - 1
    buckets.append([start, end])
  # Handle the last bucket, which might be smaller if total_samples is not divisible by bucket_size
  if total_samples % bucket_size != 0:
    start = num_buckets * bucket_size
    end = total_samples - 1
    buckets.append([start, end])
  return buckets

def get_rows(start, end):
   responses = supabase_client.table('data').select('*').gte("idx", start).lte("idx", end).execute()
   return responses


def make_json_serializable(input_dict):
    def serialize_value(value):
        if isinstance(value, list):
            # Recursively process lists
            return [serialize_value(v) for v in value]
        elif isinstance(value, dict):
            # Recursively process dictionaries
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.ndarray,)):
            # Convert numpy arrays to lists
            return value.tolist()
        return value  # Return as is if already JSON-serializable

    return {key: serialize_value(value) for key, value in input_dict.items()}

   

def upload_file(file_path, file_name):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {
        'name' : file_name,
        'parents' : [PARENT_FOLDER_ID]
    }
    file = service.files().create(
        body=file_metadata,
        media_body=file_path
    ).execute()



#upload_file("Downloaded_Songs/Jason Mraz - I Won't Give Up.mp3")
if __name__ == "__main__":
    TOTAL_SAMPLES = 10000#Sizes for testing purposes
    BUCKET_SIZE = 500 #Sizes for testing purposes
    #0. Creating buckets
    counter = 0
    bucketlist = create_buckets(TOTAL_SAMPLES, BUCKET_SIZE)
    for bucket in bucketlist:
        #1. Pulling rows from db
        print(f"----Executing bucket: {bucket}----")
        current_rows = get_rows(bucket[0], bucket[1])
        current_rows = current_rows.data
        print(f"----Pulling Data from DB----")
        if len(current_rows) == 0:
            print("ERROR IN PULLING ROWS FROM DB")
            quit()
        else:
            print(f"CURRENT ROWS\n")
        for row in current_rows:
            print(row, end = "\n")
        track_ids = [] #Store Track IDS
        for row in current_rows:
            track_ids.append(row['track_id'])
        print(f'----Retrieved Track IDs----')
        print(f'Track IDs: {track_ids}')
        print("----Starting Downloading Songs----")
        for id in track_ids:
            track_url = "https://open.spotify.com/track/" + id
            output_folder = "data_loader_spotify/Downloaded_Songs"
            download_module.download_spotify_tracks(track_url=track_url, output_folder=output_folder)
        print(f"----Tracks Successfully Downloaded----")
        print(f"----Starting Extracting Features from Tracks----")
        spotify_featureDB_columns = "rms_level , spectral_centroid , bandwidth , zero_crossing_rate , band_energy_ratio , delta_spectrum_magnitude , pitch , pitch_strength , mfcc_mean , mfcc_std , average_roughness , std_roughness , one_2hz_loudness , three_15hz_loudness , twenty_43hz_loudness , sharpness , gamma_0Hz_energy , gamma_3_15Hz_energy , gamma_20_150Hz_energy , gamma_150_1000Hz_energy"
        
        print("----CONNECTING TO spotify_features_db----")
        downloaded_files = os.listdir("data_loader_spotify/Downloaded_Songs")
        for file in downloaded_files:
            #print(f"Extracting features for : {row["track_name"]}")
            features_dict = feature_extractor_module.extract_audio_features("data_loader_spotify/Downloaded_Songs/" + file)
            features_dict["id"] = counter 
            counter += 1
            serialized_dict = make_json_serializable(features_dict)
            supabase_client.table('audio_features').insert(serialized_dict).execute()
        #Send files to db - optional
        print("----TRANSACTIONS COMPLETED SUCCESSFULLY FOR CURRENT BUCKET----")
        #Remove Songs from Downloaded Songs
        files_list = os.listdir("data_loader_spotify/Downloaded_Songs")
        for file in files_list:
            os.remove("data_loader_spotify/Downloaded_Songs/"+file)
    os.rmdir("data_loader_spotify/Downloaded_Songs")

        

