import supabase
from functools import lru_cache
import re

PROJECT_URL = "https://qbmoyulmzltkzvtqslnl.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibW95dWxtemx0a3p2dHFzbG5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI0MzI3MDMsImV4cCI6MjA0ODAwODcwM30.Kg0APL06JN3Wa4Zd7J_uDM3nOoEclpcKYOA71QYN2n8"
supabase_client = supabase.create_client(PROJECT_URL, API_KEY)

@lru_cache(maxsize=None)
def get_songs():
    song_names = supabase_client.table('data').select("track_name").execute().data
    song_list = []
    for song in song_names:
        song_list.append(song['track_name'])
    print(song_list)
    return song_list

def hamming_distance(string1: str, string2:str)->int:
    distance = 0
    string1 = re.sub('\W+','', string1).lower()
    string2 = re.sub('\W+','', string2).lower()
    for char1, char2 in zip(string1, string2):
        if (char1 != char2):
            print(f"char1:{char1} char2:{char2}")
            distance += 1
    return distance


def get_header_rows():
    header_rows = supabase_client.table('data').select("*").lte("idx", 5).execute()
    return header_rows.data

def search_results(search_string):
    suggested_songs = []
    songs = get_songs()
    for song in songs:
        hamming_dist = hamming_distance(song, search_string)
        if hamming_dist == 0:
            return [song]
        elif hamming_dist <= 3:
            suggested_songs.append(song)
        else:
            continue
    return suggested_songs

if __name__ == "__main__":
    search_result = search_results("Never gonna give up")
    print(search_result)
    print(hamming_distance("I Won't Give Up", "i wont give up"))
