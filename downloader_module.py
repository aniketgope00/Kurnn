from googleapiclient.discovery import build
from google.oauth2 import service_account
import sqlite3

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

def get_rows(start, end):
   responses = supabase_client.table('data').select('*').gte("idx", start).lte("idx", end).execute()
   return responses

if __name__ == "__main__":
    start = 500
    end = 2000
    current_rows = get_rows(start, end)
    current_rows = current_rows.data
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