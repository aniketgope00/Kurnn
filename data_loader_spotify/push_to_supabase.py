import supabase
import sqlite3

url = "https://qbmoyulmzltkzvtqslnl.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFibW95dWxtemx0a3p2dHFzbG5sIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzI0MzI3MDMsImV4cCI6MjA0ODAwODcwM30.Kg0APL06JN3Wa4Zd7J_uDM3nOoEclpcKYOA71QYN2n8"
sp_base_client = supabase.create_client(url, key)


keyList = ["index", "idx", "artist_name", "track_name", "track_id", "popularity", "year", "genre", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]

connector = sqlite3.connect("data_loader_spotify/spotify_db.sqlite3")
cursor = connector.cursor()
start = 0
end = 1
query = f"select * from DATA"
cursor.execute(query)
rows = cursor.fetchall()
for row in rows:
    print(row)

for row in rows:
    insert_row = {}
    for i, key in enumerate(keyList):
        insert_row[key] = row[i]
    print(insert_row)
    response = sp_base_client.table('data').insert(insert_row).execute()
    print(response)